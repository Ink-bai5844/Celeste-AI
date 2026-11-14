"""Bounded physical attempts through independent geometric corridor proposals.

Geometry supplies private milestones only. Every emitted input is replayed
from the actual arrival state and checked against the original public route.
"""
from dataclasses import dataclass, replace
from collections import deque
import time

from .assist_routes import (RouteProgress, advance_route_progress,
                           route_complete, start_route_progress)
from .corridor_routes import (generate_corridor_routes, FallingBlockTriggerGoal,
                              ZipMoverTriggerGoal)
from .physics import DASH, GRAB, JUMP, LEFT, RIGHT, UP, SimState
from .planner import Goal, search_leg, SearchContinuation, SearchNodeBudget
from .macro_continuation import MacroContinuation
from .search_progress import publish_corridor_progress, remember_segment


class _ConstrainedSimulator:
    def __init__(self, simulator, route, *, no_dash=False, phase=None):
        self.base = simulator
        self.map_data = simulator.map_data if hasattr(simulator, "map_data") else simulator.map
        self.route = route
        self.no_dash = no_dash
        self.phase = phase or RouteProgress()
        self._planning_identity = object()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def step(self, before, mask):
        after = self.base.step(before, mask)
        if self.no_dash and (mask & DASH or getattr(self.base, "last_step_started_dash", False)):
            return replace(after, dead=True, reason="corridor_dash_probe")
        # Each edge stops at the first public hit. Preserve its incoming
        # segment and airborne history; landing on an airborne-only branch
        # is rejected at the very step where it happens.
        progress = RouteProgress(self.phase.index,
            self.phase.left_ground or not (before.on_ground and before.vy >= 0))
        progress = advance_route_progress(self.route, progress, before, after, mask)
        if progress.violation:
            return replace(after, dead=True, reason="corridor_" + progress.violation)
        return after


@dataclass(frozen=True)
class _StageGoal(Goal):
    portal: object = None
    public_goal: object = None

    def waypoint(self, state):
        # Dynamic public exit guidance must not replace a private landing or
        # wall contact. Only the portal determines this intermediate approach.
        dynamic=getattr(self.portal,"waypoint",None)
        return dynamic(state) if callable(dynamic) else (self.portal.x,self.portal.y)

    def __getattr__(self, name):
        try:
            return getattr(self.portal, name)
        except AttributeError:
            return getattr(self.public_goal, name)

    def achieved(self, state):
        return self.public_goal.achieved(state) or self.portal.achieved(state)

    def first_hit_rejected(self, state):
        return getattr(self.public_goal, "first_hit_rejected", lambda s: False)(state)


def _opposite_grips(simulator, origin, portal):
    """Equivalent guidance choices on the other real side of a narrow shaft.

    They are private alternatives only: catching either side preserves the
    physical ascent opportunity without requiring an arbitrary wall polarity.
    Collision and all public-route conditions remain checked by the simulator.
    """
    if (portal.kind != "wall_grip" or getattr(portal,"identity","")
            or not hasattr(simulator, "_tiles")):
        return ()
    from .wall_moves import WallGripGoal, _static_faces
    room=simulator.map.rooms.get(origin.room)
    if room is None:
        return ()
    candidates=[]
    for rect,facing in _static_faces(simulator,room):
        if facing==portal.facing:
            continue
        left,top,right,bottom=rect
        x=left-4 if facing>0 else right+4
        width=(portal.x-x)*portal.facing
        if not 16<=width<=80 or not top<portal.y<bottom+11:
            continue
        if simulator.solid_at(origin.room,x,portal.y,state=origin):
            continue
        candidate=WallGripGoal("wall_grip",origin.room,x,portal.y,
            id=f"corridor_opposite:{origin.room}:{x:g}:{portal.y:g}:{facing}",
            rect=rect,facing=facing,grip_offset=portal.y-top)
        candidates.append((width,candidate))
    return tuple(row[1] for row in sorted(candidates,key=lambda row:(row[0],row[1].id))[:2])


def _support_grips(simulator, origin, portal):
    """Real exterior walls that can catch a low approach to a support.

    Landing on the top is not the only way to reach a ledge. A nearby wall
    catch can preserve a falling approach and then climb to the same support.
    These are optional physical setup edges, never substitutes for contact.
    """
    if (portal.kind != "surface" or portal.achieved(origin)
            or origin.on_ground and origin.vy >= 0
            or not hasattr(simulator, "_tiles")):
        return ()
    from .wall_moves import wall_grip_goals
    candidates = [grip for grip in wall_grip_goals(simulator, origin, portal)
        if not grip.identity and abs(grip.x-portal.x) <= 24
        and abs(grip.rect[1]-portal.y) <= 24 and not grip.achieved(origin)]
    candidates.sort(key=lambda grip:(abs(grip.rect[1]-portal.y),
        abs(grip.x-portal.x), abs(grip.x-origin.x), grip.key))
    return tuple(candidates[:2])


def _mechanism_clearance_edges(simulator, origin, goal, *, deadline,
                              max_frames, stats, max_expansions, max_edges=6,
                              destination=None, continuation=None):
    """Wait or leave a moving solid through real inputs, never by advancing time alone.

    Holding the block, letting go, and jumping away can produce different
    physical arrivals. Each branch advances the player and mechanisms together;
    a death or segment restriction rejects it before it can become guidance.
    """
    if continuation is not None:
        from .macro_continuation import MACRO_EXPANSION
        key=(id(simulator),id(stats),origin,goal,max_frames,max_edges,destination)
        def work(handle):
            if goal.kind!="mechanism_clearance" or origin.dead:
                return []
            identity=getattr(goal,"identity","")
            record=next((row for row in origin.mechanisms if row[0]==identity),None)
            away=LEFT if record is not None and origin.x<record[3] else RIGHT
            face=(RIGHT if goal.facing>0 else LEFT) if getattr(goal,"mechanism_kind","")=="zipMover" else 0
            recipes=(((),0),((),GRAB|face),((),GRAB|UP|face),
                     ((away,)*6,0),((JUMP|away,)*8,0),((JUMP|away,)*16,GRAB))
            edges=[]
            end=destination or getattr(goal,"public_goal",goal)
            for prefix,hold in recipes:
                yield MACRO_EXPANSION
                current,actions=origin,[]
                for tick in range(min(150,max_frames)):
                    if tick%16==0:
                        yield None
                    mask=prefix[tick] if tick<len(prefix) else hold
                    current=simulator.step(current,mask)
                    stats["simulated_frames"]+=1
                    actions.append(mask)
                    if current.dead:
                        break
                    if goal.achieved(current):
                        edges.append((actions,current))
                        edges.sort(key=lambda row:(-row[1].dashes,
                            (row[1].x-end.x)**2+(row[1].y-end.y)**2,
                            not (row[1].on_ground or row[1].state==1),-row[1].stamina,len(row[0])))
                        handle.results=[actions for actions,_ in edges[:max_edges]]
                        break
                    if current.room!=origin.room:
                        break
            return handle.results
        return continuation.advance(key,work,deadline=deadline,stats=stats,
            max_expansions=max_expansions,recipe_limit=6)
    if goal.kind != "mechanism_clearance" or origin.dead:
        return []
    identity=getattr(goal,"identity","")
    record=next((row for row in origin.mechanisms if row[0]==identity),None)
    away=LEFT if record is not None and origin.x<record[3] else RIGHT
    face=(RIGHT if goal.facing>0 else LEFT) if getattr(goal,"mechanism_kind","")=="zipMover" else 0
    recipes=(((),0),((),GRAB|face),((),GRAB|UP|face),
             ((away,)*6,0),((JUMP|away,)*8,0),((JUMP|away,)*16,GRAB))
    edges=[]
    for prefix,hold in recipes:
        if time.monotonic()>=deadline or stats["expansions"]>=max_expansions:
            break
        stats["expansions"]+=1
        current=origin
        actions=[]
        for tick in range(min(150,max_frames)):
            if time.monotonic()>=deadline:
                break
            mask=prefix[tick] if tick<len(prefix) else hold
            current=simulator.step(current,mask)
            stats["simulated_frames"]+=1
            actions.append(mask)
            if current.dead:
                break
            if goal.achieved(current):
                edges.append((actions,current))
                break
            if current.room!=origin.room:
                break
    # Do not let the first few neutral/falling arrivals hide a later wall
    # catch. Keep distinct physical states and prefer restored resources and
    # proximity to the actual next passage. An origin just above that passage
    # must not reverse the order and prefer a long, uncontrolled drop.
    destination=destination or getattr(goal,"public_goal",goal)
    edges.sort(key=lambda row:(-row[1].dashes,
                               (row[1].x-destination.x)**2+(row[1].y-destination.y)**2,
                               not (row[1].on_ground or row[1].state==1),
                               -row[1].stamina,len(row[0])))
    return [actions for actions,_ in edges[:max_edges]]


def search_corridor_routes(simulator, start, route, goals, *, deadline,
                           max_expansions, max_frames, variant=0, forbidden=()):
    """Search all public points continuously, with upstream arrival backtracking."""
    progress = start_route_progress(route, start)
    details = dict(corridor_attempts=[], topology={}, selected_corridor_id=None,
                   backtracks=0, corridor_guidance=True, selected_stage_corridors=[])
    if not goals or len(route["waypoints"]) != len(goals):
        details["reason"] = "unsupported_guidance_combination"
        return [], start, False, 0, 0, progress, details
    details["corridor_scheduler"] = dict(turns=0, probes=0, resumed=0, pages=0,
        routes_started=0, memory_pauses=0, max_live_nodes=0, quantum_expansions=64,
        quantum_seconds=.2, atomic_macro_seconds=.2)
    details["corridor_scheduler"].update(macro_turns=0,macro_resumed=0)
    context=dict(details=details,stats=dict(expansions=0,simulated_frames=0),cache={},
                 forbidden=set(forbidden),edges={},node_budget=SearchNodeBudget())
    work = _search_corridor_stage(simulator,start,route,goals,progress,context,
        deadline=deadline,max_expansions=max_expansions,max_frames=max_frames,variant=variant)
    try:
        while True:
            next(work)
    except StopIteration as finished:
        return finished.value
    finally:
        work.close()
        for edge in context["edges"].values():
            for handle in edge["handles"].values():
                handle.release()
            for handle in edge["macros"].values():
                handle.release()


def _search_corridor_stage(simulator,start,route,goals,progress,context,*,
                           deadline,max_expansions,max_frames,variant,prefix_actions=()):
    """A public hit continues from its exact state; suffix failure unrolls it."""
    details,stats,cache=context["details"],context["stats"],context["cache"]
    def empty(reached=False):
        return [],start,reached,stats["expansions"],stats["simulated_frames"],progress,details
    if progress.violation:
        details["reason"] = progress.violation
        return empty()
    if route_complete(route, progress):
        return empty(True)
    if time.monotonic() >= deadline or stats["expansions"]>=max_expansions or max_frames <= 0:
        details["reason"] = "corridor_budget"
        return empty()
    goal = goals[progress.index]
    corridors, inventory = [], []
    active_public_suffixes = {}
    active_private_progress = {}
    active_expansion_limit = max_expansions
    replay_dash_costs = {}
    best = ([], start, progress)
    scheduler = details["corridor_scheduler"]
    owner_root = (progress.index, replace(start, frame=0, reason=""), tuple(prefix_actions))

    def edge_context(origin, phase, portal, frames_left, owner):
        # Different geometric paths can share one exact physical edge. Keep
        # the full state, public phase and frame allowance, not the corridor's
        # cosmetic ID, so overlapping prefixes do not repeat the same work.
        key = (owner_root, origin, portal, phase, frames_left)
        edge = context["edges"].get(key)
        if edge is None:
            target = _StageGoal(portal.kind, portal.room, portal.x, portal.y,
                id=portal.id, no_dash_until=portal.no_dash_until, portal=portal, public_goal=goal)
            edge = dict(target=target, wrappers={
                True:_ConstrainedSimulator(simulator,route,no_dash=True,phase=phase),
                False:_ConstrainedSimulator(simulator,route,phase=phase)},
                handles={}, macros={}, results={}, spent={}, done=set())
            context["edges"][key] = edge
        return edge

    def search_slice(edge, origin, frames_left, no_dash, recipe, allowance):
        recipe_key = (no_dash, recipe)
        if recipe_key in edge["results"]:
            return replace(edge["results"][recipe_key],expansions=0,simulated_frames=0)
        handle = edge["handles"].setdefault(recipe_key, SearchContinuation(context["node_budget"]))
        resumed = handle.resumes
        result = search_leg(edge["wrappers"][no_dash], origin, edge["target"],
            deadline=min(deadline,time.monotonic()+.2),
            max_expansions=min(allowance,64,max(1,max_expansions//max(4,len(corridors)*4)),
                               max_expansions-stats["expansions"]),
            max_frames=frames_left, variant=recipe, allow_unsupported=False,
            continuation=handle)
        stats["expansions"] += result.expansions
        stats["simulated_frames"] += result.simulated_frames
        edge["spent"][recipe_key] = edge["spent"].get(recipe_key,0)+result.expansions
        scheduler["probes"] += 1
        scheduler["resumed"] += handle.resumes-resumed
        scheduler["max_live_nodes"] = max(scheduler["max_live_nodes"],context["node_budget"].nodes)
        if result.reached or result.reason not in ("budget", "time_budget", "expansion_budget",
                                                     "continuation_memory_budget"):
            edge["done"].add(recipe_key)
            edge["results"][recipe_key]=result
            handle.release()
        elif result.reason == "continuation_memory_budget":
            # Release this bounded recipe, then let other routes/recipes run.
            # A full shared pool must never cause an endless zero-work spin.
            edge["done"].add(recipe_key)
            scheduler["memory_pauses"] += 1
            handle.release()
        return result

    def record(identity, index, reason, actions):
        if len(details["corridor_attempts"]) < 80:
            details["corridor_attempts"].append(dict(id=identity, portal_index=index,
                public_index=progress.index,reason=reason,frames=len(actions),expanded=stats["expansions"]))

    def replay(origin, phase, actions):
        current, states, used = origin, [origin], []
        incoming_index=phase.index
        dash_starts=[]
        for mask in actions:
            # This certifies a bounded, already computed action sequence. A
            # search ending on its last quantum must still be able to prove
            # its result; no new branch or suffix is searched during replay.
            before = current
            current = simulator.step(before, mask)
            # Read the transient event immediately: held/buffered dash input
            # can start a dash without a new key edge, and goal checks may
            # perform other simulation probes.
            if getattr(simulator,"last_step_started_dash",False):
                dash_starts.append(len(used))
            stats["simulated_frames"] += 1
            used.append(mask)
            states.append(current)
            phase = advance_route_progress(route, phase, before, current, mask)
            if phase.violation:
                return None
            if phase.index>incoming_index:
                break
        remember_segment(origin, current, states)
        replay_dash_costs[(origin,tuple(used))]=tuple(dash_starts)
        return used, current, phase

    def arrivals(origin, phase, portal, local_deadline, frames_left, climb_up=False,
                 macros_only=False, following_goal=None, owner="", route_prefix=()):
        key = (replace(origin, frame=0, reason=""), portal, phase)
        saved, seen, tried_actions = [], {}, set()
        edge = edge_context(origin,phase,portal,frames_left,owner)
        target = edge["target"]
        constrained, conservative = edge["wrappers"][False], edge["wrappers"][True]

        def macro_deadline():
            # Macro families remain bounded atomic recipes, rather than
            # 64-node search slices. Their own expansion allowances stay intact,
            # but none may spend the rest of the request before another route.
            return min(local_deadline,time.monotonic()+.2)

        def macro_batches(helper, family, wrapped, allowance, **kwargs):
            # Clearance ranking also depends on the following destination;
            # equivalent contact goals with different continuations retain
            # separate macro cursors even while sharing ordinary graph work.
            handle=edge["macros"].setdefault((family,id(wrapped),tuple(sorted(kwargs.items()))),
                MacroContinuation(context["node_budget"]))
            if handle.done:
                yield list(handle.results)
                return
            while (not handle.done and time.monotonic()<local_deadline
                    and stats["expansions"]<active_expansion_limit):
                resumed=handle.resumes
                rows=helper(wrapped,origin,target,deadline=macro_deadline(),
                    max_frames=frames_left,stats=stats,
                    max_expansions=min(active_expansion_limit,
                        stats["expansions"]+max(0,allowance-handle.expansions)),
                    continuation=handle,**kwargs)
                scheduler["macro_turns"]+=1
                scheduler["macro_resumed"]+=handle.resumes-resumed
                # Non-resumable test/instrumentation adapters return a whole
                # batch without binding a continuation. Production families
                # bind it before doing physical work.
                if handle.key is None:
                    handle.done=True
                # Rank a finite internal launch family before committing its
                # first suffix. Publishing a half-enumerated batch makes CPU
                # speed choose the arrival: a slow slice may expose an early
                # spent-dash catch and postpone its better sibling behind a
                # long downstream search. Keep yielding scheduler turns while
                # enumeration resumes; only its delivery waits for completion.
                # Actual public hits and already-paid final budget results
                # remain immediately available for full-route certification.
                public_rows=[row for row in rows if isinstance(row,(tuple,list))
                    and len(row)==3 and isinstance(row[1],SimState)
                    and goal.achieved(row[1])]
                settled=(handle.done or stats["expansions"]>=active_expansion_limit
                         or time.monotonic()>=local_deadline)
                yield rows if settled else public_rows

        def admit(actions):
            if len(actions)>frames_left:
                return None
            action_key=tuple(actions)
            if action_key in tried_actions:
                return None
            # Origin, route phase, constraints and target are fixed for this
            # consumer. Replaying a cumulative macro batch again cannot change
            # its result, and rebuilding the same preview prefix is expensive.
            tried_actions.add(action_key)
            value = replay(origin, phase, actions)
            if value is None:
                return None
            used, state, next_phase = value
            if not target.achieved(state):
                return None
            if route_complete(route,next_phase):
                from .room_planning import action_fingerprint
                if (action_fingerprint((*prefix_actions,*route_prefix,*used)) in context.get("forbidden",())
                        or getattr(goals[-1],"first_hit_rejected",lambda state:False)(state)):
                    details["excluded_candidates"]=details.get("excluded_candidates",0)+1
                    return None
            # Full immutable physical state retains velocity, resources,
            # mechanisms and cassette phase; pixel-only arrivals are unsafe.
            identity = (replace(state, frame=0, reason=""), next_phase)
            if seen.get(identity, frames_left+1) <= len(used):
                return None
            seen[identity] = len(used)
            saved.append(used)
            cache[key] = saved.copy()
            return value

        def admit_batch(action_lists):
            # Atomic helpers return at most a few already simulated arrivals.
            # Certify that finite batch before yielding: if its last expansion
            # reached the public target, a preceding private arrival must not
            # hide the completed answer behind the outer budget guard.
            values=[value for actions in action_lists if (value:=admit(actions)) is not None]
            values.sort(key=lambda value:not route_complete(route,value[2]))
            return values

        # A budget-limited attempt never proves an edge impossible. Replay
        # cached successful arrivals first, then still seek new alternatives.
        for actions in cache.get(key, ()):
            if len(actions)>frames_left:
                continue
            value = admit(actions)
            if value is not None:
                yield value

        if portal.achieved(origin):
            value = admit([])
            if value is not None:
                yield value
        if portal.kind == "mechanism_clearance":
            for edges in macro_batches(_mechanism_clearance_edges,"clearance",conservative,6,
                                       destination=following_goal):
                yield from admit_batch(edges)
                yield None
        recipes=((True,4),(False,4)) if macros_only else ((True,4),(False,4),(False,0))
        if (portal.kind=="dash_block_break" and not origin.dash_attack
                and getattr(portal,"identity","") not in origin.broken):
            # Saving a dash cannot achieve an explicit unbroken-block event.
            # Try its real collision first instead of spending an 80-node
            # no-dash probe on an impossible prerequisite. The constrained
            # wrapper still enforces the user's original no-dash segment.
            recipes=tuple(row for row in recipes if not row[0])
        for no_dash, recipe in recipes:
            if time.monotonic() >= local_deadline or stats["expansions"] >= active_expansion_limit:
                break
            wrapped = conservative if no_dash else constrained
            if portal.kind=="dash_block_break" and recipe==4:
                from .dash_block_moves import dash_block_edges
                for broken in macro_batches(dash_block_edges,"dash_block",wrapped,28,
                                              max_edges=4,remember_results=False):
                    yield from admit_batch(actions for actions,_,_ in broken)
                    yield None
            if (recipe == 4 and origin.state == 1 and hasattr(simulator, "_tiles")
                    and portal.kind in {"wall_grip", "surface", "corridor", "target"}
                    and not (portal.kind=="surface" and abs(portal.x-origin.x)<=64
                             and abs(portal.y-origin.y)<=48)):
                # Nearby ledges retain the ordinary short jump/settle search.
                # A long launch macro can otherwise hold Jump through an
                # already useful landing and supply a worse suffix state.
                from .mechanism_moves import mechanism_transfer_edges
                from .wall_transfers import static_wall_transfer_edges
                from .wall_underpasses import wall_underpass_edges
                # A conditional passage can leave a real moving grip. The
                # static-wall helper deliberately rejects that source; retain
                # its actual lift-assisted transfers through the same bounded,
                # resumable recipe scheduling used by other corridor edges.
                for transfers in macro_batches(mechanism_transfer_edges,
                        "mechanism_transfer",wrapped,192,max_edges=2,remember_results=False):
                    yield from admit_batch(actions for actions,_,_ in transfers)
                    yield None
                for transfers in macro_batches(static_wall_transfer_edges,"transfer",wrapped,256,
                                                max_edges=2,remember_results=False):
                    for value in admit_batch(actions for actions,_,_ in transfers):
                        if (value[1].on_ground and value[1].vy>=0
                                and len(value[0])<frames_left and value[2].index==phase.index
                                and stats["expansions"]<active_expansion_limit
                                and time.monotonic()<local_deadline):
                            rested=admit(value[0]+[0])
                            if rested is not None:
                                yield rested
                        yield value
                    yield None
                # An upward goal may first require dropping below a ledge.
                # Every release, dash and new catch is still simulated with
                # this segment's original restrictions and full-state replay.
                for transfers in macro_batches(wall_underpass_edges,
                        "wall_underpass",wrapped,96,max_edges=2,remember_results=False):
                    yield from admit_batch(actions for actions,_,_ in transfers)
                    yield None
            if portal.kind=="refill" and recipe==4:
                from .refill_moves import refill_approach_edges
                for refills in macro_batches(refill_approach_edges,"refill",wrapped,256,
                                              max_edges=6,remember_results=False):
                    yield from admit_batch(actions for actions,_,_ in refills)
                    yield None
            if portal.kind == "wall_grip":
                from .wall_moves import wall_approach_edges, wall_relay_approach_edges
                def wall_batches():
                    found=False
                    for batch in macro_batches(wall_approach_edges,"wall",wrapped,96,max_edges=2):
                        found=found or bool(batch)
                        yield batch
                    if no_dash and recipe==4 and not found and hasattr(simulator,"_tiles"):
                        yield from macro_batches(wall_relay_approach_edges,"relay",wrapped,384,max_edges=2)
                for approaches in wall_batches():
                    if climb_up:
                        approaches.sort(key=lambda row:(row[1].y,-row[1].dashes,-row[1].stamina))
                    for value in admit_batch(actions for actions,_,_ in approaches):
                        if value[2].index>phase.index or stats["expansions"]>=active_expansion_limit:
                            yield value
                            continue
                        # A coarse grip point may be near a wall's bottom.
                        # Continue real climbing to retain useful altitude
                        # arrivals before committing the suffix to that low
                        # point. Different stamina and momentum remain distinct.
                        extended = list(value[0])
                        current = value[1]
                        higher = []
                        lowest_y=current.y
                        last_motion=0
                        last_saved_y=current.y
                        climb_deadline=macro_deadline()
                        for tick in range(min(64 if climb_up else 0, frames_left-len(extended))):
                            if time.monotonic() >= climb_deadline:
                                yield None
                                climb_deadline=macro_deadline()
                                if time.monotonic()>=local_deadline:
                                    break
                            current = wrapped.step(current, GRAB | UP)
                            stats["simulated_frames"] += 1
                            extended.append(GRAB | UP)
                            if current.dead or current.room != origin.room:
                                break
                            if current.y<lowest_y:
                                lowest_y=current.y
                                last_motion=tick
                            if tick-last_motion>=12:
                                break  # Ceiling contact cannot buy further altitude.
                            if tick in (15,31,47,63) and current.y<last_saved_y-4 and target.achieved(current):
                                candidate = admit(extended)
                                if candidate is not None:
                                    higher.append(candidate)
                                    last_saved_y=current.y
                        for candidate in reversed(higher):
                            yield candidate
                        yield value
                    yield None
            if macros_only:
                continue
            if time.monotonic() >= local_deadline or stats["expansions"] >= active_expansion_limit:
                break
            # Resume the same recipe/goal/wrapper in short physical-search
            # quanta. The original recipe allowance is cumulative, not reset
            # when another geometric route gets its turn.
            branch_expansions=80 if no_dash else 1200 if recipe==4 else 600
            recipe_key=(no_dash,recipe)
            completed=edge["results"].get(recipe_key)
            if completed is not None and completed.reached:
                # Admission is per route prefix (notably its blacklist). A
                # different consumer may still use this exact physical result
                # even if its predecessor could not publish it to the cache.
                value=admit(completed.actions)
                if value is not None:
                    yield value
            while (recipe_key not in edge["done"] and time.monotonic()<local_deadline
                    and stats["expansions"]<active_expansion_limit
                    and edge["spent"].get(recipe_key,0)<branch_expansions):
                result=search_slice(edge,origin,frames_left,no_dash,recipe,
                    branch_expansions-edge["spent"].get(recipe_key,0))
                if not result.reached:
                    yield None
                    if result.expansions==0 and result.reason not in ("time_budget","budget"):
                        break
                    continue
                value=admit(result.actions)
                if value is None:
                    break
                if (value[1].on_ground and value[1].vy>=0 and len(value[0])<frames_left
                        and value[2].index==phase.index and stats["expansions"]<active_expansion_limit
                        and time.monotonic()<local_deadline):
                    rested=admit(value[0]+[0])
                    if rested is not None:
                        yield rested
                yield value
                yield None
                break
            handle=edge["handles"].get(recipe_key)
            if handle is not None:
                handle.release()
            edge["done"].add(recipe_key)

    def make_visit(corridor):
        portals=tuple(corridor.portals)+(goal,)
        visited={}

        def edge_arrivals(*args,**kwargs):
            return arrivals(*args,**kwargs,owner=corridor.id)

        def visit(index,origin,phase,prefix,prepared_index=None):
            # Only currently active, physically reached milestones earn this
            # preference. Coarse fractions avoid rewarding a path merely for
            # splitting the same space into more private guidance points.
            levels=active_private_progress.setdefault(corridor.id,[])
            band=(min(3,4*index//len(portals)) if phase.index==progress.index
                  and 0<index<len(portals) else 0)
            levels.append(band)
            try:
                return (yield from visit_arrival(index,origin,phase,prefix,prepared_index))
            finally:
                levels.remove(band)

        def visit_arrival(index,origin,phase,prefix,prepared_index=None):
            nonlocal best
            if route_complete(route,phase):
                from .room_planning import action_fingerprint
                if (action_fingerprint((*prefix_actions,*prefix)) in context.get("forbidden",())
                        or getattr(goals[-1],"first_hit_rejected",lambda state:False)(origin)):
                    details["excluded_candidates"]=details.get("excluded_candidates",0)+1
                    return None
                return prefix,origin,phase
            if phase.index>progress.index:
                levels=active_public_suffixes.setdefault(corridor.id,[])
                levels.append(phase.index)
                try:
                    suffix=yield from _search_corridor_stage(simulator,origin,route,goals,phase,context,
                        deadline=deadline,max_expansions=max_expansions,
                        max_frames=max_frames-len(prefix),variant=variant,
                        prefix_actions=(*prefix_actions,*prefix))
                finally:
                    levels.remove(phase.index)
                combined=prefix+suffix[0]
                if (suffix[5].index,len(combined))>(best[2].index,len(best[0])):
                    best=combined,suffix[1],suffix[5]
                if suffix[2]:
                    return combined,suffix[1],suffix[5]
                return None
            if (index>=len(portals) or len(prefix)>=max_frames or time.monotonic()>=deadline
                    or stats["expansions"]>=max_expansions):
                return None
            identity=(index,replace(origin,frame=0,reason=""),phase,prepared_index)
            if visited.get(identity,max_frames+1)<=len(prefix):
                return None
            visited[identity]=len(prefix)
            conditional=(portals[index].kind in {"mechanism_clearance","dash_approach","dash_block_break"} or
                         isinstance(portals[index],(FallingBlockTriggerGoal,ZipMoverTriggerGoal)) or
                         portals[index].kind=="corridor" and getattr(portals[index],"identity",""))
            targets=([index] if conditional else list(range(index,min(len(portals),index+2))))
            if (len(targets)==2 and portals[index].kind=="wall_grip"
                    and not getattr(portals[index],"identity","")
                    and portals[index+1].kind=="refill" and portals[index+1].y>origin.y+8):
                targets.reverse()
            if (len(targets)==2 and origin.state==1 and portals[index].kind=="corridor"
                    and portals[index+1].kind=="surface"):
                targets.reverse()

            def proposals():
                # Keep the existing physical wall/refill/setup families and
                # their ordering. Every failed search quantum also yields.
                if (len(targets)==2 and all(portals[i].kind=="wall_grip" and
                        not getattr(portals[i],"identity","") for i in targets)
                        and portals[targets[0]].facing==portals[targets[1]].facing
                        and portals[targets[1]].y<portals[targets[0]].y-8):
                    choices=[]
                    for target_index in targets:
                        portal=portals[target_index]
                        following=next((p for p in portals[target_index+1:]
                                        if p.kind!="mechanism_clearance"),portals[-1])
                        for value in edge_arrivals(origin,phase,portal,deadline,max_frames-len(prefix),
                                climb_up=following.y<portal.y-8,macros_only=True,route_prefix=prefix):
                            if value is None:
                                yield None
                                continue
                            actions,arrival,next_phase=value
                            if route_complete(route,next_phase):
                                yield target_index+1,target_index,actions,arrival,next_phase,"wall_alternative"
                                continue
                            starts=replay_dash_costs[(origin,tuple(actions))]
                            priority=(len(starts),-arrival.dashes,-arrival.stamina,arrival.y,
                                      -target_index,-starts[0] if starts else 0)
                            choices.append((priority,target_index,actions,arrival,next_phase))
                    for _,target_index,actions,arrival,next_phase in sorted(choices,key=lambda row:row[0]):
                        yield target_index+1,target_index,actions,arrival,next_phase,"wall_alternative"
                for target_index in targets:
                    portal=portals[target_index]
                    found=False
                    choices=(portal,)+_opposite_grips(simulator,origin,portal)
                    following=next((p for p in portals[target_index+1:]
                                    if p.kind!="mechanism_clearance"),portals[-1])
                    # A support catch prepares this particular milestone. Its
                    # continuation must now try that milestone, rather than
                    # recursively generating another catch for the same one.
                    # Otherwise opposite wall catches can consume every frame
                    # while never trying the ledge between them. All original
                    # physical recipes remain available, and failure returns
                    # to the parent's other full-state catch alternatives.
                    if prepared_index==target_index:
                        details["support_setup_target_attempts"]=details.get("support_setup_target_attempts",0)+1
                        support_grips=()
                    else:
                        support_grips=_support_grips(simulator,origin,portal)
                    for grip in support_grips:
                        for value in edge_arrivals(origin,phase,grip,deadline,max_frames-len(prefix),
                                climb_up=True,macros_only=True,following_goal=portal,route_prefix=prefix):
                            if value is None:
                                yield None
                                continue
                            actions,arrival,next_phase=value
                            yield target_index,target_index,actions,arrival,next_phase,"support_wall_approach"
                    for option in choices:
                        for value in edge_arrivals(origin,phase,option,deadline,max_frames-len(prefix),
                                climb_up=following.y<option.y-8,following_goal=following,route_prefix=prefix):
                            if value is None:
                                yield None
                                continue
                            found=True
                            actions,arrival,next_phase=value
                            yield target_index+1,target_index,actions,arrival,next_phase,"physical_arrival"
                    if not found:
                        record(corridor.id,target_index,"physical_leg_budget",prefix)

            # Keep the current arrival's continuation on this corridor's
            # backtracking stack. Each physical quantum still yields to other
            # corridors, but extra portals must not exponentially divide a
            # deep branch's share at every ancestor. Bounded recipes eventually
            # fail back to the waiting alternative arrivals.
            work=deque([(proposals(),None)])
            public_turn=0
            while work and time.monotonic()<deadline and stats["expansions"]<max_expansions:
                # A real public hit opens another complete stage search. It
                # must not monopolize this arrival's parent for the remaining
                # request: another velocity/resource state at the same public
                # point may be the one that can finish the route. Prioritize
                # these suffixes three turns in four, but keep a turn for the
                # parent's alternatives. Private-only recursion keeps its
                # original depth-first order rather than losing time at every
                # intermediate portal.
                public_positions=[position for position,(_,item) in enumerate(work)
                                  if item is not None and item[2]]
                public_pending=bool(public_positions)
                selected=0
                if public_pending:
                    selected=public_positions[0]
                    if public_turn%4==3:
                        selected=next((position for position,(_,item) in enumerate(work)
                                       if item is None or not item[2]),selected)
                    public_turn+=1
                work.rotate(-selected)
                iterator,metadata=work.popleft()
                work.rotate(selected)
                turn_started=time.monotonic()
                turn_expansions=stats["expansions"]
                finished=None
                while True:
                    try:
                        value=next(iterator)
                    except StopIteration as stopped:
                        finished=stopped
                        break
                    # A cheap prioritized landing/resource suffix may finish
                    # in a few macro checks. Give it one small physical-work
                    # quantum, rather than switching on bookkeeping alone.
                    if (metadata is not None and value is None
                            and stats["expansions"]-turn_expansions<8
                            and time.monotonic()-turn_started<.02):
                        continue
                    break
                if finished is not None:
                    if metadata is not None:
                        if finished.value is not None:
                            return finished.value
                        details["backtracks"]+=1
                        target_index,combined,_=metadata
                        record(corridor.id,target_index,"suffix_failed_backtrack",combined)
                    continue
                if metadata is None or metadata[2] or public_pending:
                    work.append((iterator,metadata))
                else:
                    work.appendleft((iterator,metadata))
                if metadata is None and value is not None:
                    next_index,target_index,actions,arrival,next_phase,label=value
                    combined=prefix+actions
                    if (next_phase.index,len(combined))>(best[2].index,len(best[0])):
                        best=combined,arrival,next_phase
                    record(corridor.id,target_index,label,combined)
                    if route_complete(route,next_phase):
                        from .room_planning import action_fingerprint
                        if (action_fingerprint((*prefix_actions,*combined)) not in context.get("forbidden",())
                                and not getattr(goals[-1],"first_hit_rejected",lambda state:False)(arrival)):
                            return combined,arrival,next_phase
                        details["excluded_candidates"]=details.get("excluded_candidates",0)+1
                    else:
                        prepared=(target_index if label=="support_wall_approach" else None)
                        work.appendleft((visit(next_index,arrival,next_phase,combined,prepared),
                                         (target_index,combined,next_phase.index>phase.index)))
                yield None
            return None
        return visit(0,start,progress,[])

    next_offset=0
    pages_done=False
    generation_pending=False
    generation_retries=0
    generation_continuation=None
    retry_after_turn=0
    unstarted=deque()
    pending=deque()
    selection_turn=0
    def load_page():
        nonlocal next_offset,pages_done,generation_pending,generation_retries,retry_after_turn,generation_continuation
        if pages_done or next_offset>=24 or time.monotonic()>=deadline:
            pages_done=True
            return
        now=time.monotonic()
        generation_slice=(min(2.,max(.05,(deadline-now)*.2)) if not corridors else
                          min(2.,.25*(2**min(generation_retries,3))))
        continuation_args=({"continuation":generation_continuation}
                           if generation_continuation is not None else {})
        page=generate_corridor_routes(simulator,start,goal,max_routes=6,
            candidate_offset=next_offset,deadline=min(deadline,now+generation_slice),
            **continuation_args)
        generation_continuation=getattr(page,"continuation",generation_continuation)
        next_offset=getattr(page,"next_offset",next_offset+len(page))
        known={corridor.id for corridor in corridors}
        fresh=[corridor for corridor in page if corridor.id not in known]
        complete=getattr(page,"complete",True)
        generation_pending=not complete and next_offset<24
        if fresh:
            generation_retries=0
        if next_offset>=24 or complete and (not fresh or len(page)<6):
            pages_done=True
        elif not fresh:
            # Empty time-limited discovery is not an exhausted inventory.
            # Let known paths get a round, then retry the same saved cursor.
            generation_retries+=1
            retry_after_turn=sum(row["turns"] for row in inventory)+max(1,len(pending))
            pages_done=True
        details["generation_pending"]=generation_pending
        details["generation_reason"]=getattr(page,"reason","bounded_inventory_complete")
        scheduler["pages"]+=1
        for corridor in fresh:
            position=len(corridors)
            corridors.append(corridor)
            inventory.append(dict(id=corridor.id,status="pending",turns=0,expansions=0,
                conditions=corridor.to_dict().get("conditions",[]),
                reason=(f"需完成 {len(corridor.conditional_on)} 个机关条件，并验证通道畅通"
                        if getattr(corridor,"conditional_on",()) else ""),
                points=[dict(room=corridor.room,x=x,y=y) for x,y in corridor.points[:60]]))
            unstarted.append(position)
        details["topology"]=dict(active_id=None,corridors=inventory,public_index=progress.index)
        publish_corridor_progress(inventory)

    def completed(position,answer):
        corridor=corridors[position]
        details["selected_corridor_id"]=corridor.id
        details["selected_stage_corridors"].insert(0,dict(public_index=progress.index,id=corridor.id))
        details["reason"]="corridor_route_found"
        details["topology"]=dict(active_id=corridor.id,corridors=inventory,public_index=progress.index)
        publish_corridor_progress(inventory,active_id=corridor.id)
        return (*answer[:2],True,stats["expansions"],stats["simulated_frames"],answer[2],details)

    load_page()
    while (unstarted or pending or not pages_done or generation_pending) and time.monotonic()<deadline and stats["expansions"]<max_expansions:
        if (generation_pending and pages_done and not unstarted
                and (not pending or sum(row["turns"] for row in inventory)>=retry_after_turn)):
            pages_done=False
        if not unstarted and not pages_done:
            load_page()
        if unstarted:
            # A genuine tiny first search, on the same saved no-dash recipe
            # used later, gives every generated route an opportunity before
            # one difficult wall macro or recursive suffix spends the budget.
            position=unstarted.popleft()
            corridor=corridors[position]
            row=inventory[position]
            portal=(tuple(corridor.portals)+(goal,))[0]
            row.update(status="trying",reason="首次短探测")
            details["topology"]=dict(active_id=corridor.id,corridors=inventory,public_index=progress.index)
            publish_corridor_progress(inventory,active_id=corridor.id,throttle_seconds=.1)
            before=stats["expansions"]
            edge=edge_context(start,progress,portal,max_frames,corridor.id)
            result=search_slice(edge,start,max_frames,True,4,
                min(4,max(1,(max_expansions-stats["expansions"])//max(1,len(unstarted)+1))))
            starter_answer=None
            if result.reached:
                value=replay(start,progress,result.actions)
                if value is not None and edge["target"].achieved(value[1]):
                    key=(replace(start,frame=0,reason=""),portal,progress)
                    cache.setdefault(key,[]).append(value[0])
                    if route_complete(route,value[2]):
                        from .room_planning import action_fingerprint
                        if (action_fingerprint((*prefix_actions,*value[0])) not in context.get("forbidden",())
                                and not getattr(goals[-1],"first_hit_rejected",lambda state:False)(value[1])):
                            starter_answer=value
                        else:
                            details["excluded_candidates"]=details.get("excluded_candidates",0)+1
            row["turns"]+=1
            row["expansions"]+=stats["expansions"]-before
            row.update(status="pending",reason="已探测，等待下一轮")
            scheduler["turns"]+=1
            scheduler["routes_started"]+=1
            if starter_answer is not None:
                row.update(status="trying",reason="完整路线已找到，等待逐帧复验")
                return completed(position,starter_answer)
            pending.append((position,make_visit(corridor)))
            yield None
            continue
        if not pending:
            if generation_pending:
                yield None
                continue
            break
        # A route that has physically completed a public waypoint should be
        # able to finish its suffix. Uniformly nesting K-way rotation at every
        # waypoint would divide useful suffix work by K**depth. Keep one FIFO
        # turn in four for all paths, after every new path's initial probe.
        selected=0
        if selection_turn%4!=3:
            levels=[]
            for position,_ in pending:
                identity=corridors[position].id
                public=max(active_public_suffixes.get(identity,()),default=progress.index)
                # Local milestone progress does not establish a useful arrival
                # for later public points. Ordered requests retain the proven
                # public-phase preference and otherwise rotate their paths.
                private=(max(active_private_progress.get(identity,()),default=0)
                         if len(route["waypoints"])==1 and public==progress.index else 0)
                levels.append((public,private))
            selected=max(range(len(levels)),key=lambda index:levels[index])
        selection_turn+=1
        pending.rotate(-selected)
        position,work=pending.popleft()
        pending.rotate(selected)
        corridor=corridors[position]
        row=inventory[position]
        row.update(status="trying",reason="继续已保存的搜索")
        details["topology"]=dict(active_id=corridor.id,corridors=inventory,public_index=progress.index)
        publish_corridor_progress(inventory,active_id=corridor.id,throttle_seconds=.1)
        before=stats["expansions"]
        try:
            next(work)
        except StopIteration as finished:
            row["turns"]+=1
            row["expansions"]+=stats["expansions"]-before
            scheduler["turns"]+=1
            if finished.value is not None:
                return completed(position,finished.value)
            if stats["expansions"]>=max_expansions or time.monotonic()>=deadline:
                row.update(status="pending",reason="本次预算已用完，尚未充分搜索")
            else:
                row.update(status="failed",reason="本轮候选已试完，尚未找到连续可行的动作")
        else:
            row["turns"]+=1
            row["expansions"]+=stats["expansions"]-before
            row.update(status="pending",reason="已保存进度，等待下一轮")
            scheduler["turns"]+=1
            pending.append((position,work))
        publish_corridor_progress(inventory,active_id=corridor.id,throttle_seconds=.1)
        yield None
    for position,_ in pending:
        inventory[position].update(status="pending",reason="预算已用完，搜索进度尚未穷尽")
    details["topology"]=dict(active_id=None,corridors=inventory,public_index=progress.index)
    details["reason"]=("corridor_generation_time_budget" if generation_pending else
                       "no_corridor_route_within_budget" if corridors else "no_geometric_corridors")
    publish_corridor_progress(inventory)
    return best[0],best[1],False,stats["expansions"],stats["simulated_frames"],best[2],details
