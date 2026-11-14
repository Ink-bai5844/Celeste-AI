"""Continuous lift-assisted transfers from the real side of a zip mover.

Ride timing is supplied by zip_grip_edges. This module only emits real inputs
from that state; it never relocates the player or advances a mechanism alone.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time

from .physics import DASH, DOWN, GRAB, JUMP, LEFT, RIGHT, UP, _truth
from .search_progress import remember_segment
from .search_resources import grip_resource_band
from .zip_grips import _contact, zip_side_grip_goals


def _airborne_break_approach_recipes(origin, target):
    """Short flights to a nearby face; landing/recovery remains simulated."""
    if (origin.room != target.room or origin.state != 0 or origin.on_ground
            or abs(target.x-origin.x)>192 or abs(target.y-origin.y)>160):
        return
    toward = RIGHT if target.x>origin.x else LEFT if target.x<origin.x else 0
    # Preserve momentum or catch a real wall before spending another dash.
    yield (0,)*140, "air_coast"
    yield (GRAB|toward,)*140, "air_grab"
    if origin.dashes<=0:
        return
    # Delaying the dash lets gravity carry the body below an overhang. The
    # remaining horizontal flight can then land on a lower recovery support.
    # All timings are alternatives, not assumed reachability certificates.
    aims = tuple(dict.fromkeys(aim for aim in (toward,UP|toward,DOWN|toward) if aim))
    for coast in (0,8,16,24,32):
        release = max(coast,int(bool(origin.prev_input&DASH)))
        for aim in aims:
            for settle in dict.fromkeys((toward,0)):
                yield ((toward,)*release+(DASH|aim,)*6+(settle,)*100,
                       f"air_coast_dash:{coast}:{aim}:{settle}")


def active_dash_block_edges(simulator, origin, target, *, deadline,
                            max_frames=220, stats=None, max_expansions=None,
                            max_edges=4):
    """Physically approach and break an explicit useful intermediate block.

    The resulting rebound may continue anywhere: passing immediately through
    the old block footprint is not required. The caller still replays the
    complete public route and all its original segment restrictions.
    """
    from .corridor_routes import DashBlockApproachGoal
    from .corridor_planning import _StageGoal
    from .dash_block_moves import dash_block_edges
    from .planner import search_leg
    if (origin.dead or max_frames<=0 or max_edges<=0 or time.monotonic()>=deadline
            or getattr(simulator,"no_dash",False)):
        return []
    identity=getattr(target,"identity","")
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    room=data.rooms.get(origin.room)
    entity=next((e for e in room.entities if f"{room.name}:{e.id}"==identity),None) if room else None
    if entity is None or entity.kind!="dashBlock" or not _truth(entity.attrs.get("canDash",True)):
        return []
    stats=stats if stats is not None else {}
    limit=min(max_expansions if max_expansions is not None else float("inf"),
              stats.get("expansions",0)+320)
    if stats.get("expansions",0)>=limit:
        return []
    public=getattr(target,"public_goal",None)
    left,top,right,bottom=target.rect
    # Side contact is an interval, not one arbitrary airborne pixel. A real
    # landing beside the block can be a better launch than its guide height.
    bounds=((target.x-4,top+1,target.x+4,bottom+10) if target.dash_dx else
            (math.nextafter(left-4,math.inf),target.y-4,
             math.nextafter(right+4,-math.inf),target.y+4))
    approach=DashBlockApproachGoal("dash_approach",target.room,target.x,target.y,
        id="active_approach:"+target.key,identity=identity,
        bounds=bounds)
    guided=(_StageGoal(approach.kind,approach.room,approach.x,approach.y,
                      id=approach.id,portal=approach,public_goal=public)
            if public is not None else approach)
    results,seen=[],set()

    def public_hit(state):
        return (public is not None and public.achieved(state)
                and not getattr(public,"first_hit_rejected",lambda state:False)(state))

    def retain(actions,state,label):
        key=tuple(actions)
        if key not in seen:
            seen.add(key)
            results.append((actions,state,label))

    def break_from(actions,state,approach_label=""):
        if time.monotonic()>=deadline or stats.get("expansions",0)>=limit:
            return
        # An airborne approach can reach the face just before a recovery
        # landing. Restore a dash only through actual safe idle updates; do
        # not manufacture a resource or discard a useful zero-dash arrival.
        if state.dashes<=0 and not state.dash_attack:
            stats["expansions"]=stats.get("expansions",0)+1
            actions=list(actions)
            for _ in range(min(60,max_frames-len(actions))):
                if time.monotonic()>=deadline:
                    return
                state=simulator.step(state,0)
                stats["simulated_frames"]=stats.get("simulated_frames",0)+1
                actions.append(0)
                if state.dead:
                    return
                if public_hit(state):
                    retain(actions,state,"active_dash_block:public_hit")
                    return
                if state.dashes>0 or state.dash_attack:
                    break
                if state.room!=origin.room:
                    return
        for suffix,final,label in dash_block_edges(simulator,state,target,deadline=deadline,
                max_frames=max_frames-len(actions),stats=stats,max_expansions=limit,
                max_edges=max_edges,remember_results=False):
            retain(actions+suffix,final,"active_"+label+
                   (":via_"+approach_label if approach_label else ""))

    if identity in origin.broken:
        return [([],origin,"active_dash_block:already_broken")]
    # The short collision family can also handle an approach a few pixels
    # outside the guidance patch. The physics must still reach and hit it.
    close=(abs(origin.x-target.x)<=48 and top<origin.y and origin.y-11<bottom
           if target.dash_dx else
           abs(origin.y-target.y)<=48 and origin.x+4>left and origin.x-4<right)
    if close:
        break_from([],origin)
        if results:
            return results[:max_edges]

    class NoDashApproach:
        no_dash=True
        def __init__(self,base):
            self.base=base
            self._planning_identity=object()
        def __getattr__(self,name):
            return getattr(self.base,name)
        def step(self,before,mask):
            after=self.base.step(before,mask)
            if mask&DASH or getattr(self.base,"last_step_started_dash",False):
                return replace(after,dead=True,reason="active_break_approach_dash")
            return after

    ordinary=NoDashApproach(simulator)
    if stats.get("expansions",0)<limit:
        # A rebound can settle and walk along a support before its next break.
        # This cheap ordinary approach preserves that natural continuation
        # instead of requiring the graph to rediscover every walking frame.
        stats["expansions"]=stats.get("expansions",0)+1
        state,actions=origin,[]
        direction=RIGHT if target.x>=origin.x else LEFT
        for _ in range(min(160,max_frames)):
            if time.monotonic()>=deadline:
                break
            state=ordinary.step(state,direction)
            stats["simulated_frames"]=stats.get("simulated_frames",0)+1
            actions.append(direction)
            if state.dead:
                break
            if public_hit(state):
                retain(actions,state,"active_dash_block:public_hit")
                break
            if identity in state.broken:
                retain(actions,state,"active_dash_block:approach_break")
                break
            if approach.achieved(state):
                break_from(actions,state)
                break
            if state.room!=origin.room:
                break
        if results:
            return results[:max_edges]

    for wrapped,quota in ((ordinary,96),(simulator,160)):
        if time.monotonic()>=deadline or stats.get("expansions",0)>=limit:
            break
        if wrapped is simulator:
            # The broad fallback must not be the only way to discover a
            # midair coast -> dash -> real landing -> second dash sequence.
            # Keep ordinary/no-dash approaches first and share the same total
            # 320-expansion allowance with every flight and collision recipe.
            for recipe,label in _airborne_break_approach_recipes(origin,target):
                if time.monotonic()>=deadline or stats.get("expansions",0)>=limit:
                    break
                stats["expansions"]=stats.get("expansions",0)+1
                state,actions=origin,[]
                for mask in recipe[:max_frames]:
                    if time.monotonic()>=deadline:
                        break
                    state=simulator.step(state,mask)
                    stats["simulated_frames"]=stats.get("simulated_frames",0)+1
                    actions.append(mask)
                    if state.dead:
                        break
                    if public_hit(state):
                        retain(actions,state,"active_dash_block:"+label+":public_hit")
                        break
                    if identity in state.broken:
                        retain(actions,state,"active_dash_block:"+label+":approach_break")
                        break
                    if state.room!=origin.room:
                        break
                    if approach.achieved(state):
                        break_from(actions,state,label)
                        break
                if results:
                    return results[:max_edges]
            if time.monotonic()>=deadline or stats.get("expansions",0)>=limit:
                break
        result=search_leg(wrapped,origin,guided,deadline=deadline,
            max_expansions=min(quota,limit-stats.get("expansions",0)),
            max_frames=max_frames,variant=4,allow_unsupported=False)
        stats["expansions"]=stats.get("expansions",0)+result.expansions
        stats["simulated_frames"]=stats.get("simulated_frames",0)+result.simulated_frames
        if not result.reached or result.state.dead:
            continue
        if public_hit(result.state):
            retain(result.actions,result.state,"active_dash_block:public_hit")
        elif approach.achieved(result.state):
            break_from(result.actions,result.state)
        if results:
            break
    return results[:max_edges]


def _airborne_grip_recipes(origin, target):
    """Brake outside the moving face before the spring's upward flight ends."""
    x, _ = target.waypoint(origin)
    toward = RIGHT if x >= origin.x else LEFT
    away = LEFT if toward == RIGHT else RIGHT
    face = RIGHT if target.facing > 0 else LEFT
    catch = GRAB | UP | face
    for wait in (0, 4, 8, 12, 16, 24):
        yield (0,) * wait + (catch,) * 140, f"air_catch:{wait}"
    for approach in (6, 8, 9, 10, 12, 16, 20):
        for brake in (0, 4, 6, 8, 10):
            for coast in (0, 4, 8):
                yield ((toward,) * approach + (away,) * brake + (0,) * coast
                       + (catch,) * 140, f"air_brake:{approach}:{brake}:{coast}")


def _transfer_recipes(origin, target, allow_dash):
    waypoint = getattr(target, "waypoint", None)
    x, y = waypoint(origin) if waypoint is not None else (target.x, target.y)
    toward = RIGHT if x >= origin.x else LEFT
    away = LEFT if origin.facing > 0 else RIGHT
    face = (RIGHT if target.facing > 0 else LEFT) if target.kind == "wall_grip" else toward
    catch = GRAB | UP | face
    held = GRAB | (RIGHT if origin.facing > 0 else LEFT)
    release = (held,) if origin.prev_input & JUMP else ()
    # Away jumps inherit horizontal lift. A neutral climb-jump preserves a
    # rising mover's vertical boost without crossing its unsafe top surface.
    launches = (away, 0)
    for direction in launches:
        for ticks in (12, 20, 8, 28):
            for after in (catch, GRAB | face, 0):
                yield (release + (JUMP | direction,) * ticks + (after,) * 140,
                       f"jump:{direction}:{ticks}:{after}")
    # Native ClimbJump also accepts motion toward the held face. This keeps
    # the lift-assisted rise while crossing a narrow mover's top; the caller's
    # simulator still decides whether attached spikes/ceilings allow it.
    # Keep the original away/neutral family first and bound these additions so
    # all old dash recipes remain reachable under the 192-expansion quota.
    climb = GRAB | UP | (RIGHT if origin.facing > 0 else LEFT)
    for hold in (0, 8, 16):
        prefix = (climb,) * hold if hold else release
        for ticks in (12, 20, 28, 32, 40):
            for after in ((catch, GRAB | face) if hold == 0 else (catch,)):
                yield (prefix + (JUMP | toward,) * ticks + (after,) * 140,
                       f"toward_jump:{hold}:{toward}:{ticks}:{after}")
    # Triggering a mover can open a route below it. Release and let normal
    # gravity/contact decide whether the jump happens in air, from a wall,
    # or through the ordinary ground buffer. Any rest/refill is physical.
    # This small family shares the existing quota without displacing old
    # away/neutral dash recipes, including their downward-target variants.
    for wait in (8, 16, 20, 24, 32):
        for ticks in (12, 20, 28):
            yield ((0,) * wait + (JUMP | toward,) * ticks + (catch,) * 140,
                   f"release_jump:{wait}:{toward}:{ticks}:{catch}")
    if not allow_dash or origin.dashes <= 0:
        return
    aims = (UP | toward, toward, UP)
    if y > origin.y + 32:
        aims += (DOWN | toward,)
    for delay in (20, 28, 12, 36, 8, 16, 24, 4):
        for direction in launches:
            for aim in aims:
                for after in (catch, GRAB | face):
                    # Hold aim through the first two DashCoroutine updates.
                    yield (release + (JUMP | direction,) * delay
                           + (DASH | aim,) * 6 + (after,) * 140,
                           f"jump_dash:{direction}:{delay}:{aim}:{after}")
    # Later toward launches complement the preserved original dash order.
    # They do not change the spring/airborne-catch family or any ride phase.
    for delay in (28, 36, 20):
        for hold in (0, 8, 16):
            prefix = (climb,) * hold if hold else release
            for aim in (UP | toward, toward, UP):
                yield (prefix + (JUMP | toward,) * delay
                       + (DASH | aim,) * 6 + (catch,) * 140,
                       f"toward_jump_dash:{hold}:{toward}:{delay}:{aim}:{catch}")


def mechanism_transfer_edges(simulator, origin, target, *, deadline,
                             max_frames=220, stats=None, max_expansions=None,
                             max_edges=4, continuation=None, remember_results=True):
    """Return exact ``(actions, state, label)`` transfers from actual zip grips.

    Targets may be dynamic grips, supports, or a caller's public-hit wrapper.
    Only its achieved predicate accepts an arrival. Deadline, frame budget and
    the caller's wrapped Simulator apply to every step and every branch.
    """
    if continuation is not None:
        return _continued_mechanism_transfers(simulator, origin, target,
            deadline=deadline, max_frames=max_frames, stats=stats,
            max_expansions=max_expansions, max_edges=max_edges,
            continuation=continuation, remember_results=remember_results)
    if (origin.dead or max_frames <= 0 or max_edges <= 0
            or time.monotonic() >= deadline):
        return []
    stats = stats if stats is not None else {}
    if max_expansions is not None and stats.get("expansions", 0) >= max_expansions:
        return []
    grips = ([grip for grip in zip_side_grip_goals(simulator, origin, target)
              if _contact(origin, grip)] if origin.state == 1 else [])
    airborne = (not grips and origin.state != 1 and not origin.on_ground and target.kind == "wall_grip"
                and getattr(target,"mechanism_kind","") == "zipMover")
    if not grips and not airborne:
        return []
    source_identity = grips[0].identity if grips else "airborne"
    if target.achieved(origin):
        return [([], origin, f"mechanism_transfer:{source_identity}:already_reached:dash_starts=0")]
    seen, recipes_seen = {}, set()
    recipes = (_airborne_grip_recipes(origin,target) if airborne else
               _transfer_recipes(origin,target,not getattr(simulator,"no_dash",False)))
    for recipe, label in recipes:
        if (time.monotonic() >= deadline or max_expansions is not None
                and stats.get("expansions", 0) >= max_expansions):
            break
        recipe = recipe[:max_frames]
        if recipe in recipes_seen:
            continue
        recipes_seen.add(recipe)
        stats["expansions"] = stats.get("expansions", 0) + 1
        state, actions, states, dash_starts = origin, [], [origin], []
        for mask in recipe:
            if time.monotonic() >= deadline:
                break
            state = simulator.step(state, mask)
            if getattr(simulator,"last_step_started_dash",False):
                dash_starts.append(len(actions))
            stats["simulated_frames"] = stats.get("simulated_frames", 0) + 1
            actions.append(mask)
            states.append(state)
            if state.dead:
                break
            if target.achieved(state):
                key = replace(state, frame=0, reason="")
                rank = (len(dash_starts), -state.dashes, state.y, state.vy,
                        -state.stamina, len(actions), tuple(-tick for tick in dash_starts))
                if key not in seen or rank < seen[key][0]:
                    seen[key] = (rank, actions, state, label, states)
                break
            if state.room != origin.room:
                break
    chosen = _mechanism_transfer_rows(seen.values(), max_edges)
    result = []
    for rank, actions, state, label, states in chosen:
        if remember_results:
            remember_segment(origin,state,states)
        result.append((actions,state,
                       f"mechanism_transfer:{source_identity}:{label}:dash_starts={rank[0]}"))
    return result


def _mechanism_transfer_rows(rows, max_edges):
    chosen, deferred, bands = [], [], set()
    for row in sorted(rows, key=lambda row:row[0]):
        rank, _, state, _, _ = row
        band = (rank[0], state.dashes, state.state, round(state.y/8),
                round(state.vx/45), round(state.vy/45), grip_resource_band(state.stamina))
        if band in bands:
            deferred.append(row)
        else:
            chosen.append(row)
            bands.add(band)
        if len(chosen) >= max_edges:
            break
    chosen.extend(deferred[:max(0,max_edges-len(chosen))])
    return sorted(chosen, key=lambda row:row[0])


def _continued_mechanism_transfers(simulator, origin, target, *, deadline,
                                    max_frames, stats, max_expansions, max_edges,
                                    continuation, remember_results):
    from .macro_continuation import MACRO_EXPANSION
    if stats is None:
        stats = continuation.__dict__.setdefault("stats", {})
    key = (id(simulator), id(stats), origin, target, max_frames, max_edges,
           remember_results)

    def recipes(work):
        if origin.dead or max_frames <= 0 or max_edges <= 0:
            return []
        grips = ([grip for grip in zip_side_grip_goals(simulator, origin, target)
                  if _contact(origin, grip)] if origin.state == 1 else [])
        airborne = (not grips and origin.state != 1 and not origin.on_ground
                    and target.kind == "wall_grip"
                    and getattr(target, "mechanism_kind", "") == "zipMover")
        if not grips and not airborne:
            return []
        source_identity = grips[0].identity if grips else "airborne"
        if target.achieved(origin):
            return [([], origin,
                     f"mechanism_transfer:{source_identity}:already_reached:dash_starts=0")]
        seen, recipes_seen = {}, set()
        candidates = (_airborne_grip_recipes(origin, target) if airborne else
                      _transfer_recipes(origin, target, not getattr(simulator, "no_dash", False)))
        for recipe, label in candidates:
            recipe = recipe[:max_frames]
            if recipe in recipes_seen:
                continue
            recipes_seen.add(recipe)
            yield MACRO_EXPANSION
            state, actions, states, dash_starts = origin, [], [origin], []
            for mask in recipe:
                state = simulator.step(state, mask)
                if getattr(simulator, "last_step_started_dash", False):
                    dash_starts.append(len(actions))
                stats["simulated_frames"] = stats.get("simulated_frames", 0) + 1
                actions.append(mask)
                states.append(state)
                if state.dead:
                    break
                if target.achieved(state):
                    identity = replace(state, frame=0, reason="")
                    rank = (len(dash_starts), -state.dashes, state.y, state.vy,
                            -state.stamina, len(actions), tuple(-tick for tick in dash_starts))
                    if identity not in seen or rank < seen[identity][0]:
                        # Keep only the current rollout's full history. Other
                        # candidates retain bounded action prefixes and exact
                        # terminal states for the caller's own certification.
                        seen[identity] = (rank, actions, state, label, ())
                        if remember_results:
                            remember_segment(origin, state, states)
                        work.results = [(masks, end,
                            f"mechanism_transfer:{source_identity}:{name}:dash_starts={order[0]}")
                            for order, masks, end, name, _ in
                            _mechanism_transfer_rows(seen.values(), max_edges)]
                    break
                if state.room != origin.room:
                    break
                if len(actions) % 16 == 0:
                    yield None
        return work.results

    recipe_limit = continuation.__dict__.setdefault("recipe_limit",
        max(0, max_expansions - stats.get("expansions", 0))
        if max_expansions is not None else None)
    return continuation.advance(key, recipes, deadline=deadline, stats=stats,
        max_expansions=max_expansions, recipe_limit=recipe_limit)
