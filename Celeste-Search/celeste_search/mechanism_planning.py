"""Bounded search over continuous, physically reached mechanism events.

Spring launches, moving-side contacts, ride phases and permanent landings are
alternative search states. None of their positions or resources are invented.
The ordinary ordered-route replay remains the sole execution admission check.
"""
from dataclasses import dataclass, replace
import heapq
import math
import time

from .assist_routes import (advance_route_progress, route_complete,
                           start_route_progress)
from .corridor_planning import _ConstrainedSimulator, _StageGoal
from .planner import search_leg
from .search_progress import remember_segment
from .search_resources import grip_resource_band


@dataclass
class _Node:
    state: object
    actions: list
    phase: object
    stages: tuple = ()
    dashes: int = 0
    targets: object = None
    cursor: int = 0
    direct_probed: bool = False
    near_probed: bool = False
    ride_sampled: tuple = ()
    fallback_targets: object = None
    fallback_cursor: int = 0
    base_priority: float = 0.
    contact_key: str = ""
    contact_identity: str = ""
    contact_history: tuple = ()


def _trim_frontier(frontier, nodes, width=32):
    """Reserve different contacts before adding their finer timing variants."""
    if width <= 0:
        return []
    groups, extras, bands = {}, [], set()
    for entry in sorted(frontier):
        child = nodes[entry[1]]
        state = child.state
        phase_record = next((record for record in state.mechanisms
                             if record[0] == child.contact_identity), None)
        motion = (phase_record[1], phase_record[2]//8) if phase_record else None
        band = (child.phase.index, child.contact_key, state.dashes, state.state,
                state.facing, round(state.y/16), grip_resource_band(state.stamina),
                motion, round(state.lift_vx/60), round(state.lift_vy/60), state.lift_timer,
                state.broken)
        if band in bands:
            extras.append(entry)
        else:
            bands.add(band)
            motion_group = (motion[0], round(state.y/48)) if motion else None
            groups.setdefault((child.phase.index, child.contact_key, state.broken, motion_group), []).append(entry)
    retained = []
    for index in range(max((len(group) for group in groups.values()), default=0)):
        for group in groups.values():
            if index < len(group):
                retained.append(group[index])
                if len(retained) >= width:
                    return retained
    return retained + extras[:max(0, width-len(retained))]


def _zip_launch_windows(rides, limit=12):
    """Bound held/climbing phase representatives without losing early exits."""
    unique = {}
    for row in rides:
        unique.setdefault(tuple(row[0]), row)
    rows = list(unique.values())
    chosen = []
    def add(row):
        if row is not None and all(row is not old for old in chosen):
            chosen.append(row)
    for row in rows:
        if row[2].endswith(":goal"):
            add(row)
    if rows:
        add(rows[0])
    for token in ("outgoing8", "outgoing16", "outgoing24", "ride"):
        add(next((row for row in rows if row[2].endswith(":"+token)
                  and ":cycle1:" not in row[2]), None))
    # Interleave physical phases and height variants. One mode's several
    # nearby timings must not replace the other mode's higher real launch.
    for token in ("return24", "return72", "return_end"):
        for climbing in (False, True):
            add(next((row for row in rows if row[2].endswith(":"+token)
                      and (":return_climb:" in row[2]) == climbing), None))
    for climbing in (True, False):
        add(next((row for row in rows if row[2].endswith(":outgoing16")
                  and ":cycle1:" in row[2]
                  and (":return_climb:" in row[2]) == climbing), None))
    return chosen[:limit]


def search_mechanism_routes(simulator, start, route, goals, *, deadline,
                            max_expansions, max_frames, variant=0, use_detours=True, forbidden=()):
    """Keep alternative arrivals and their entire mechanism/route state."""
    from .mechanism_routes import held_zip_contacts, relay_candidates, relay_priority
    from .wall_moves import wall_approach_edges
    from .wall_transfers import static_wall_transfer_edges
    from .wall_underpasses import wall_underpass_edges
    from .wall_descents import wall_descent_edges
    from .zip_grips import zip_grip_edges
    from .mechanism_moves import mechanism_transfer_edges, active_dash_block_edges
    from .step_cache import ExactStepCache
    from .detour_guidance import StaticDetourGuide
    simulator = ExactStepCache(simulator)
    detours = StaticDetourGuide(simulator, deadline=deadline) if use_detours else None
    from .room_planning import action_fingerprint
    blacklist = set(forbidden)
    progress = start_route_progress(route, start)
    stats = dict(expansions=0, simulated_frames=0)
    details = dict(mechanism_guidance=True, nodes=1, stages=[], attempts=[], frontier_trace=[], landing_attempts=[],
                   near_public_probes=[])
    root = _Node(start, [], progress)
    best = root
    if not goals or len(goals) != len(route["waypoints"]):
        details["reason"] = "invalid_public_goals"
        return [], start, False, 0, 0, progress, details
    if detours is not None:
        detours.prepare(start, goals[0])
    prepared_phases = {0}
    nodes = [root]
    frontier = [(0., 0)]
    seen = {(replace(start, frame=0, reason=""), progress): [(0, 0)]}

    def available():
        return time.monotonic() < deadline and stats["expansions"] < max_expansions

    def record(node, target, result, frames):
        if len(details["attempts"]) < 160:
            details["attempts"].append(dict(public_index=node.phase.index,
                target=target.key, kind=target.kind, result=result, frames=frames,
                expansions=stats["expansions"], depth=len(node.stages)))

    def replay(node, actions):
        state, phase, used, states, dashes = node.state, node.phase, [], [node.state], 0
        for mask in actions[:max_frames-len(node.actions)]:
            # Certify this already computed, frame-bounded prefix even if its
            # last physical recipe used the final search time slice. This
            # starts no new search and preserves every public restriction.
            before = state
            state = simulator.step(before, mask)
            dashes += bool(getattr(simulator, "last_step_started_dash", False))
            stats["simulated_frames"] += 1
            used.append(mask)
            states.append(state)
            phase = advance_route_progress(route, phase, before, state, mask)
            if state.dead or phase.violation:
                return None
            if phase.index > node.phase.index:
                break
        remember_segment(node.state, state, states)
        return used, state, phase, dashes

    def near_public_actions(node, goal, trigger):
        """One small ordinary exact suffix from a genuine nearby arrival."""
        source_room = getattr(goal, "source_room", "") or goal.room
        x, y = getattr(goal, "tile_x", goal.x), getattr(goal, "tile_y", goal.y)
        if (node.near_probed or not available() or len(node.actions) >= max_frames
                or node.state.room != source_room
                or math.hypot(node.state.x-x, node.state.y-y) > 96):
            return None
        node.near_probed = True
        wrapped = _ConstrainedSimulator(simulator, route, no_dash=True, phase=node.phase)
        result = search_leg(wrapped, node.state, goal, deadline=deadline,
            max_expansions=min(96, max_expansions-stats["expansions"]),
            max_frames=max_frames-len(node.actions), variant=4, allow_unsupported=False)
        stats["expansions"] += result.expansions
        stats["simulated_frames"] += result.simulated_frames
        if len(details["near_public_probes"]) < 96:
            details["near_public_probes"].append(dict(trigger=trigger, public_index=node.phase.index,
                x=node.state.x, y=node.state.y, frames=len(node.actions), dashes_used=node.dashes,
                reached=result.reached, expansions=result.expansions))
        if result.reached:
            return result.actions
        record(node, goal, "near_public_no_dash_budget", len(result.actions))
        return None

    def push(node, actions, target, label, *, ride=False):
        nonlocal best
        value = replay(node, actions)
        if value is None:
            return None
        used, state, phase, dashes = value
        if not used or (phase.index == node.phase.index and not ride and not target.achieved(state)):
            return None
        total = node.actions + used
        if route_complete(route, phase) and action_fingerprint(total) in blacklist:
            details["excluded_candidates"] = details.get("excluded_candidates", 0) + 1
            return None
        child = _Node(state, total, phase, node.stages + (label,), node.dashes+dashes)
        child.contact_history = node.contact_history + (target.key,)
        if getattr(target, "mechanism_kind", "") == "zipMover" and phase.index == node.phase.index:
            # The caller samples the full bounded cycle for this arrival.
            # Descendant windows must not recursively wait another whole cycle.
            child.ride_sampled = (getattr(target, "identity", ""),)
        key = (replace(state, frame=0, reason=""), phase)
        cost = (child.dashes, len(total))
        arrivals = seen.get(key, [])
        # A shorter prefix has more frames left even when it spent a dash.
        # Retain both tradeoffs; lexicographic dash preference can otherwise
        # discard the only continuation that fits the total frame budget.
        if any(dashes <= cost[0] and frames <= cost[1] for dashes,frames in arrivals):
            return None
        seen[key] = [(dashes,frames) for dashes,frames in arrivals
                     if not (cost[0] <= dashes and cost[1] <= frames)] + [cost]
        record(node, target, label, len(used))
        if route_complete(route, phase):
            return child
        current_goal = goals[phase.index]
        # A newly opened world must receive one actual continuation before
        # its spent-dash cost competes with cheaper old-world contacts. This
        # changes no ranking and grants no success: only replaying the exact
        # original public suffix can advance its phase or complete the route.
        if state.broken-node.state.broken:
            suffix = near_public_actions(child, current_goal, "opened_world")
            if suffix is not None:
                completed = push(child, suffix, current_goal, "public:opened_world_no_dash")
                if completed is not None:
                    return completed
        if detours is not None and phase.index not in prepared_phases:
            detours.prepare(state, current_goal)
            prepared_phases.add(phase.index)
        # A real recovery landing is a useful boundary between mechanism
        # transfers and ordinary platform routes. Try a bounded continuation
        # immediately, before optimistic ride states monopolize the frontier.
        # Its input prefix is replayed against the original public phase below.
        if (target.kind == "surface" and state.on_ground and state.dashes > 0
                and len(child.stages) >= 2 and not route["segments"][phase.index].get("no_ground")
                and len(details["landing_attempts"]) < 3 and available()):
            from .corridor_planning import search_corridor_routes
            suffix_route = dict(waypoints=route["waypoints"][phase.index:],
                                segments=route["segments"][phase.index:])
            suffix = search_corridor_routes(simulator,state,suffix_route,goals[phase.index:],
                deadline=min(deadline,time.monotonic()+8.0),
                max_expansions=min(1000,max_expansions-stats["expansions"]),
                max_frames=max_frames-len(total),variant=variant)
            stats["expansions"] += suffix[3]
            stats["simulated_frames"] += suffix[4]
            details["landing_attempts"].append(dict(x=state.x,y=state.y,
                frames=len(total),reached=suffix[2],expansions=suffix[3]))
            if suffix[2]:
                completed = push(child,suffix[0],current_goal,"landing_corridor")
                if completed is not None:
                    return completed
        score = relay_priority(simulator, state, current_goal)
        if detours is not None:
            score += detours.score(state, current_goal)
        score += child.dashes*12 + len(total)*.04 + max(0.,50-state.stamina)*.15
        score += max(0, 3-grip_resource_band(state.stamina)[1])*1.5
        # Revisiting a contact at another real phase remains legal, but must
        # compete with unexplored continuations instead of dominating them.
        # Count events, not ride samples belonging to the same arrival.
        score += node.contact_history.count(target.key)*16
        # relay_priority already accounts for the actual lift in its launch
        # envelope. Rewarding it a second time makes repeated rides outrank
        # their useful departures, even as they consume stamina and frames.
        score -= phase.index*10000
        child.base_priority = score
        child.contact_key = target.key
        child.contact_identity = getattr(target,"identity","")
        index = len(nodes)
        nodes.append(child)
        if len(details["frontier_trace"])<128:
            details["frontier_trace"].append(dict(index=index,score=score,target=target.key,
                label=label,frames=len(total),dashes_used=child.dashes,
                x=state.x,y=state.y,vx=state.vx,vy=state.vy,dashes=state.dashes,
                stamina=state.stamina,lift_vx=state.lift_vx,lift_vy=state.lift_vy,
                state=state.state,public_index=phase.index))
        heapq.heappush(frontier, (score, index))
        if (phase.index, -score) > (best.phase.index, -getattr(best, "score", math.inf)):
            best = child
            best.score = score
        return None

    def expand_target(node, portal, *, public=False, macros_only=False, search_only=False):
        """Yield only actual goal contacts; collect different physical arrivals."""
        origin = node.state
        phase = node.phase
        goal = goals[phase.index]
        target = _StageGoal(portal.kind, portal.room, portal.x, portal.y,
            id=portal.id, no_dash_until=portal.no_dash_until,
            portal=portal, public_goal=goal)
        frames_left = max_frames-len(node.actions)
        restrictions = route["segments"][phase.index]
        seen_actions = set()
        macro_bands = {}
        any_macro = False
        dash_break=portal.kind=="dash_block_break"
        passes=(() if restrictions.get("no_dash") else (False,)) if dash_break else (
            (True,) if restrictions.get("no_dash") else (True, False))
        for no_dash in passes:
            if not available():
                break
            wrapped = _ConstrainedSimulator(simulator, route, no_dash=no_dash, phase=phase)
            candidates = []
            if dash_break and not search_only:
                candidates.extend(active_dash_block_edges(wrapped,origin,target,
                    deadline=deadline,max_frames=frames_left,stats=stats,
                    max_expansions=min(max_expansions,stats["expansions"]+320),max_edges=4))
            # These four generators gate dash recipes on the original dash
            # count. With no buffered dash, their relaxed pass repeats the
            # exact conservative inputs. Descent can collect a refill before
            # choosing its launch, and generic search can recover resources;
            # both must still receive the relaxed pass.
            repeat_macros = not no_dash and origin.dashes <= 0 and origin.dash_buffer <= 0
            # The same physical launch may reach a private contact or the
            # public exit. A relay onto a fixed wall must retain that wall's
            # jump capabilities rather than falling back to a fresh approach.
            if not dash_break and not repeat_macros and not search_only and (not public or origin.state == 1):
                candidates.extend(mechanism_transfer_edges(wrapped, origin, target,
                    deadline=deadline, max_frames=frames_left, stats=stats,
                    max_expansions=min(max_expansions,stats["expansions"]+192),max_edges=4))
            if not dash_break and not search_only and origin.state == 1 and hasattr(simulator, "_tiles"):
                if not repeat_macros:
                    candidates.extend(static_wall_transfer_edges(wrapped, origin, target,
                        deadline=deadline, max_frames=frames_left, stats=stats,
                        max_expansions=min(max_expansions,stats["expansions"]+210),max_edges=4))
                    candidates.extend(wall_underpass_edges(wrapped, origin, target,
                        deadline=deadline, max_frames=frames_left, stats=stats,
                        max_expansions=min(max_expansions,stats["expansions"]+96),max_edges=4))
                candidates.extend(wall_descent_edges(wrapped, origin, target,
                    deadline=deadline, max_frames=frames_left, stats=stats,
                    max_expansions=min(max_expansions,stats["expansions"]+96),max_edges=4))
            if not repeat_macros and not public and not search_only:
                if portal.kind == "wall_grip":
                    candidates.extend(wall_approach_edges(wrapped,origin,target,
                        deadline=deadline,max_frames=frames_left,stats=stats,
                        max_expansions=min(max_expansions,stats["expansions"]+96),max_edges=3))
            if candidates:
                any_macro = True
            # Interleave launch families before retaining a small arrival
            # beam. Near-identical catches each generate an entire set of
            # ride windows; they must not crowd out sibling contacts.
            families = {}
            for candidate in candidates:
                families.setdefault(candidate[2].split(":",1)[0],[]).append(candidate)
            candidates = [family[index] for index in range(max(
                (len(family) for family in families.values()),default=0))
                for family in families.values() if index<len(family)]
            retained_arrivals = 0
            for actions, arrival, label in candidates:
                identity = tuple(actions)
                if identity in seen_actions:
                    continue
                seen_actions.add(identity)
                # Near-identical run-ups should not each multiply into every
                # ride window. Keep resource, contact and mechanism phases
                # distinct; full-state replay/dedup remains authoritative.
                band = (arrival.room, arrival.state, arrival.facing, arrival.dashes,
                    grip_resource_band(arrival.stamina),round(arrival.x/8),round(arrival.y/8),
                    round(arrival.vx/45),round(arrival.vy/45),
                    round(arrival.lift_vx/60),round(arrival.lift_vy/60),arrival.lift_timer,
                    arrival.broken,
                    tuple((row[0],row[1],row[2]//8,round(row[3]/8),round(row[4]/8))
                          for row in arrival.mechanisms))
                if band in macro_bands and macro_bands[band] <= len(actions):
                    continue
                macro_bands[band] = len(actions)
                if portal.kind=="surface" and len(actions)<frames_left:
                    yield actions+[0], label+":support_update"
                yield actions, label
                retained_arrivals += 1
                if retained_arrivals >= max(4,len(families)):
                    break
            if any_macro or macros_only:
                continue
            # A spring's no-dash side contact can need a full ascent and
            # descent. Preserve this resource branch before relaxing it.
            spring_contact = (origin.auto_jump_active and portal.kind=="wall_grip"
                              and getattr(portal,"mechanism_kind","")=="zipMover")
            quota = (1000 if spring_contact else 160) if no_dash else (900 if public else 400)
            result = search_leg(wrapped,origin,target,deadline=deadline,
                max_expansions=min(quota,max_expansions-stats["expansions"]),
                max_frames=frames_left,variant=4,allow_unsupported=False)
            stats["expansions"] += result.expansions
            stats["simulated_frames"] += result.simulated_frames
            if result.reached and tuple(result.actions) not in seen_actions:
                seen_actions.add(tuple(result.actions))
                if portal.kind=="surface" and len(result.actions)<frames_left:
                    yield result.actions+[0], "physical_contact:support_update"
                yield result.actions, "physical_contact"

    def extend_zip(node, used, arrival, phase, portal, goal):
        """One bounded real return, with held and safe-side climb variants."""
        wrapped = _ConstrainedSimulator(simulator, route,
            no_dash=bool(route["segments"][phase.index].get("no_dash")), phase=phase)
        ride_stats = dict(expansions=0, simulated_frames=0,
                          max_expansions=max_expansions-stats["expansions"])
        rides = []
        for climbing in (False, True):
            rides.extend(zip_grip_edges(wrapped, arrival, portal, goal, deadline=deadline,
                max_frames=min(280, max_frames-len(node.actions)-len(used)),
                stats=ride_stats, cycle_windows=True, return_climb=climbing))
        stats["expansions"] += ride_stats["expansions"]
        stats["simulated_frames"] += ride_stats["simulated_frames"]
        for ride_actions, _, ride_label in _zip_launch_windows(rides):
            answer = push(node, used+ride_actions, portal, portal.key+":"+ride_label, ride=True)
            if answer is not None:
                return answer
        return None

    answer = None
    # The frontier is bounded below. Historical arrivals already pruned from
    # it should not end the search before its actual work/time budget is used.
    while frontier and available() and len(nodes) < 2048:
        priority, index = heapq.heappop(frontier)
        node = nodes[index]
        if route_complete(route, node.phase):
            answer = node
            break
        if len(node.actions) >= max_frames or len(node.stages) >= 16:
            continue
        goal = goals[node.phase.index]
        source_room = getattr(goal,"source_room","") or goal.room
        target_x = getattr(goal,"tile_x",goal.x)
        target_y = getattr(goal,"tile_y",goal.y)
        if node.fallback_targets is None:
            node.fallback_targets = []
        # A real rebound/refill arrival may already be one ordinary movement
        # away from the public point. Normal airborne states have no direct
        # mechanism macro, so give this short exact suffix a turn before all
        # private contacts. The original goal and complete state stay intact.
        if not node.direct_probed:
            previous_count = len(nodes)
            suffix = near_public_actions(node, goal, "frontier")
            if suffix is not None:
                answer = push(node, suffix, goal, "public:near_no_dash")
                if answer is not None:
                    break
            if len(nodes) > previous_count:
                # Continue the newly reached public phase first, while
                # retaining the old node for genuine suffix backtracking.
                node.direct_probed = True
                node.fallback_targets.append((goal, True))
                heapq.heappush(frontier, (node.base_priority+8, index))
                continue
        # Do not repeatedly spend a direct search on a distant summit before
        # a spring or mover has supplied the height needed for its final leg.
        if (not node.direct_probed and node.state.room==source_room and node.state.y-target_y <= 136
                and math.hypot(node.state.x-target_x,node.state.y-target_y) <= 248):
            for actions,label in expand_target(node,goal,public=True,macros_only=True):
                answer = push(node,actions,goal,"public:"+label)
                if answer is not None:
                    break
            if answer is not None:
                break
            node.fallback_targets.append((goal, True))
        node.direct_probed = True
        # Diagnostic labels contain motion details, not contact-goal keys.
        # Exact full-state deduplication controls revisits; a mover can be
        # useful again at another actual phase and is never blacklisted here.
        if node.targets is None:
            # Normal relay ranking excludes already achieved contacts. A new
            # root can nevertheless be holding a zip at a useful real phase;
            # sample that one bounded cycle without fabricating an approach
            # or consuming a slot reserved for a different relay target.
            for portal in held_zip_contacts(simulator,node.state):
                if portal.identity in node.ride_sampled:
                    continue
                node.ride_sampled += (portal.identity,)
                answer = extend_zip(node, [], node.state, node.phase, portal, goal)
                if answer is not None:
                    break
            if answer is not None:
                break
            node.targets = relay_candidates(simulator,node.state,goal,limit=6)
            if route["segments"][node.phase.index].get("no_dash"):
                node.targets=[target for target in node.targets if target.kind!="dash_block_break"]
            node.targets.sort(key=lambda candidate:node.contact_history.count(candidate.key))
        # Resume sibling contacts through the frontier, giving a useful
        # physical arrival a chance to continue before probing every sibling.
        # Cheap capability probes run before generic searches. A failed
        # optimistic target must not consume a full leg search at every ride
        # sample while an early departure or backward relay waits untouched.
        fallback = node.cursor >= len(node.targets)
        if fallback:
            candidates = node.fallback_targets[node.fallback_cursor:node.fallback_cursor+1]
            node.fallback_cursor += len(candidates)
        else:
            candidates = [(portal, False) for portal in node.targets[node.cursor:node.cursor+1]]
            node.cursor += len(candidates)
        for portal, public in candidates:
            if not available():
                break
            if portal.achieved(node.state):
                identity = getattr(portal, "identity", "")
                if (getattr(portal, "mechanism_kind", "") == "zipMover"
                        and identity not in node.ride_sampled):
                    node.ride_sampled += (identity,)
                    answer = extend_zip(node, [], node.state, node.phase, portal, goal)
                    if answer is not None:
                        break
                continue
            found = False
            for actions,label in expand_target(node,portal,public=public,
                    macros_only=not fallback,search_only=fallback):
                found = True
                answer = push(node,actions,portal,portal.key+":"+label)
                if answer is not None:
                    break
                # Retain multiple real moving-platform launch windows,
                # including early windows before a later collision/return.
                if getattr(portal,"mechanism_kind","")=="zipMover":
                    value = replay(node,actions)
                    if value is None or value[2].index!=node.phase.index:
                        continue
                    used, arrival, phase, _ = value
                    answer = extend_zip(node, used, arrival, phase, portal, goal)
                if answer is not None:
                    break
            if not found:
                record(node,portal,"no_contact_within_budget" if fallback else "no_macro_contact",0)
                if not fallback:
                    node.fallback_targets.append((portal,public))
            if answer is not None:
                break
        if answer is not None:
            break
        if node.cursor < len(node.targets):
            heapq.heappush(frontier,(node.base_priority+8+2*node.cursor,index))
        elif node.fallback_cursor < len(node.fallback_targets):
            heapq.heappush(frontier,(node.base_priority+48+4*node.fallback_cursor,index))
        # Preserve different mechanisms, lift windows and resource states,
        # rather than keeping only the closest coordinates to the exit.
        if len(frontier)>32:
            frontier=_trim_frontier(frontier,nodes)
            heapq.heapify(frontier)
    final=answer or best
    details.update(nodes=len(nodes),stages=list(final.stages),
                   reason="mechanism_route_found" if answer else "mechanism_budget",
                   guidance_profile="fixed_terrain_progress" if use_detours else "mechanism_capability",
                   detour_guidance=detours.cache_info() if detours is not None else dict(enabled=False),
                   step_cache=simulator.cache_info())
    return (final.actions,final.state,answer is not None,stats["expansions"],
            stats["simulated_frames"],final.phase,details)
