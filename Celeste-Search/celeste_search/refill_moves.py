"""Bounded, physically replayed approaches to an individual airborne refill.

Refills restore a charge without cancelling momentum or the dash cooldown.
Keep several actual pickup states so a relay can continue from a high arrival
instead of committing to the first, possibly fast-falling contact.
"""
from __future__ import annotations

from dataclasses import replace
import time

from .physics import DASH, DOWN, JUMP, LEFT, RIGHT, UP
from .search_progress import remember_segment


def _recipes(origin, goal, allow_dash):
    waypoint = getattr(goal, "waypoint", None)
    x, y = waypoint(origin) if waypoint is not None else (goal.x, goal.y)
    toward = RIGHT if x >= origin.x else LEFT
    away = LEFT if toward == RIGHT else RIGHT
    vertical = UP if y <= origin.y else DOWN
    grounded = origin.on_ground and origin.vy >= 0
    release = (0,) if origin.prev_input & JUMP else ()

    # Coasting and braking are useful for descending pickups without spending
    # the charge needed for the next leg. They also work from zero charges.
    yield (toward,) * 90, "coast"
    yield (0,) * 90, "fall"
    for ticks in (8, 16, 24, 32, 48):
        yield (toward,) * ticks + (0,) * 70, f"coast_release:{ticks}"
        yield ((toward,) * ticks + (away,) * 4 + (0,) * 66,
               f"coast_brake:{ticks}")
    if grounded:
        for ticks in (12, 20, 28, 36):
            for after in (toward, 0):
                yield (release + (JUMP | toward,) * ticks + (after,) * 90,
                       f"jump:{ticks}:{after}")
    if not allow_dash or not (origin.dashes > 0 or grounded):
        return

    # Keep the aim through DashBegin and its following update: a one-frame
    # diagonal followed by horizontal movement can become a horizontal dash.
    if grounded:
        # A distant pickup may require using the rest of the current ledge
        # before takeoff. These are real walking prefixes, not relocated
        # launch states: leaving the support too early, spikes, and lost
        # coyote time all remain ordinary simulated failures. Keep a small
        # set ahead of the denser stationary-launch timing variants.
        if abs(x-origin.x)>96:
            for runup in (16,32,8,24):
                for ticks in (20,28):
                    for aim in (vertical | toward,toward):
                        for after in (toward,0):
                            yield ((toward,)*runup + release
                                   + (JUMP | toward,)*ticks
                                   + (DASH | aim,)*6 + (after,)*90,
                                   f"runup_jump_dash:{runup}:{ticks}:{aim}:{after}")
        aims = (vertical, vertical | toward)
        for ticks in (24, 20, 28, 16, 32, 12):
            for coast in (0, 4, 8):
                for aim in aims:
                    for after in (vertical, toward, 0):
                        yield (release + (JUMP | toward,) * ticks + (JUMP,) * coast
                               + (DASH | aim,) * 6 + (after,) * 90,
                               f"jump_dash:{ticks}:{coast}:{aim}:{after}")
    else:
        aims = (vertical | toward, toward)
        if abs(x - origin.x) < 24:
            aims = (vertical, *aims)
        for delay in (12, 16, 8, 4, 0, 20, 24, 2):
            for wait in (toward, 0):
                for aim in aims:
                    for after in (toward, 0):
                        yield ((wait,) * delay + (DASH | aim,) * 6 + (after,) * 90,
                               f"air_dash:{delay}:{wait}:{aim}:{after}")


def refill_approach_edges(simulator, origin, goal, *, deadline, max_frames=180,
                          stats=None, max_expansions=None, max_edges=6, continuation=None,
                          remember_results=True):
    """Return exact ``(actions, state, label)`` refill contacts from ``origin``.

    Every frame goes through the supplied simulator, including constraint
    wrappers. The goal's own predicate is authoritative; proximity to its
    centre never substitutes for the requested entity's actual pickup.
    Expansion limits use the caller's cumulative ``stats`` just like wall edges.
    """
    if continuation is not None:
        return _continued_refills(simulator,origin,goal,deadline=deadline,
            max_frames=max_frames,stats=stats,max_expansions=max_expansions,
            max_edges=max_edges,continuation=continuation,remember_results=remember_results)
    if (origin.dead or origin.room != goal.room or goal.kind != "refill"
            or max_frames <= 0 or max_edges <= 0 or time.monotonic() >= deadline):
        return []
    stats = stats if stats is not None else {}
    if max_expansions is not None and stats.get("expansions", 0) >= max_expansions:
        return []
    if goal.achieved(origin):
        return [([], origin, "refill:already_contacting:dash_starts=0")]
    candidates, seen, recipes_seen = [], {}, set()
    allow_dash = not getattr(simulator, "no_dash", False)
    for recipe, label in _recipes(origin, goal, allow_dash):
        if (time.monotonic() >= deadline or max_expansions is not None
                and stats.get("expansions", 0) >= max_expansions):
            break
        recipe = recipe[:max_frames]
        if recipe in recipes_seen:
            continue
        recipes_seen.add(recipe)
        stats["expansions"] = stats.get("expansions", 0) + 1
        current, used, states, dash_starts = origin, [], [origin], []
        for mask in recipe:
            if time.monotonic() >= deadline:
                break
            current = simulator.step(current, mask)
            # Read the transient event before another recipe can overwrite it.
            if getattr(simulator, "last_step_started_dash", False):
                dash_starts.append(len(used))
            stats["simulated_frames"] = stats.get("simulated_frames", 0) + 1
            used.append(mask)
            states.append(current)
            if current.dead or current.room != origin.room:
                break
            if goal.achieved(current):
                key = replace(current, frame=0, reason="")
                rank = (len(dash_starts), -current.dashes, current.y, current.vy,
                        -current.stamina, len(used), tuple(-i for i in dash_starts))
                previous = seen.get(key)
                if previous is None or rank < previous[0]:
                    row = (rank, used, current, label, states)
                    seen[key] = row
                break

    candidates = sorted(seen.values(), key=lambda row: row[0])
    chosen, deferred, bands = [], [], set()
    for row in candidates:
        rank, _, state, _, _ = row
        # Separate rising, hovering and falling contacts, different pickup
        # heights, horizontal momentum, and whether a dash is still in flight.
        band = (rank[0], state.dashes, round(state.y / 8), round(state.vy / 45),
                round(state.vx / 45), state.state, bool(state.dash_frames),
                bool(state.dash_cooldown), int(state.stamina / 20))
        if band in bands:
            deferred.append(row)
        else:
            bands.add(band)
            chosen.append(row)
        if len(chosen) >= max_edges:
            break
    chosen.extend(deferred[:max(0, max_edges - len(chosen))])
    chosen.sort(key=lambda row: row[0])
    result = []
    for rank, actions, state, label, states in chosen:
        if remember_results:
            remember_segment(origin, state, states)
        result.append((actions, state, f"refill:{label}:dash_starts={rank[0]}"))
    return result


def _refill_results(rows,max_edges):
    chosen,deferred,bands=[],[],set()
    for row in sorted(rows,key=lambda row:row[0]):
        rank,_,state,_,_=row
        band=(rank[0],state.dashes,round(state.y/8),round(state.vy/45),
              round(state.vx/45),state.state,bool(state.dash_frames),
              bool(state.dash_cooldown),int(state.stamina/20))
        if band in bands:
            deferred.append(row)
        else:
            chosen.append(row)
            bands.add(band)
        if len(chosen)>=max_edges:
            break
    chosen.extend(deferred[:max(0,max_edges-len(chosen))])
    return [(actions,state,f"refill:{label}:dash_starts={rank[0]}")
            for rank,actions,state,label,_ in sorted(chosen,key=lambda row:row[0])]


def _continued_refills(simulator,origin,goal,*,deadline,max_frames,stats,
                       max_expansions,max_edges,continuation,remember_results):
    from .macro_continuation import MACRO_EXPANSION
    if stats is None:
        stats=continuation.__dict__.setdefault("stats",{})
    key=(id(simulator),id(stats),origin,goal,max_frames,max_edges)
    def recipes(work):
        if (origin.dead or origin.room!=goal.room or goal.kind!="refill"
                or max_frames<=0 or max_edges<=0):
            return []
        if goal.achieved(origin):
            return [([],origin,"refill:already_contacting:dash_starts=0")]
        seen,recipes_seen={},set()
        allow_dash=not getattr(simulator,"no_dash",False)
        for recipe,label in _recipes(origin,goal,allow_dash):
            recipe=recipe[:max_frames]
            if recipe in recipes_seen:
                continue
            recipes_seen.add(recipe)
            yield MACRO_EXPANSION
            current,used,states,dash_starts=origin,[],[origin],[]
            for mask in recipe:
                if len(used)%16==0:
                    yield None
                current=simulator.step(current,mask)
                if getattr(simulator,"last_step_started_dash",False):
                    dash_starts.append(len(used))
                stats["simulated_frames"]=stats.get("simulated_frames",0)+1
                used.append(mask)
                states.append(current)
                if current.dead or current.room!=origin.room:
                    break
                if goal.achieved(current):
                    identity=replace(current,frame=0,reason="")
                    rank=(len(dash_starts),-current.dashes,current.y,current.vy,
                          -current.stamina,len(used),tuple(-i for i in dash_starts))
                    previous=seen.get(identity)
                    if previous is None or rank<previous[0]:
                        seen[identity]=(rank,used,current,label,())
                        if len(seen)>64:
                            chosen_ids={replace(row[1],frame=0,reason="")
                                        for row in _refill_results(seen.values(),max_edges*4)}
                            ranked=sorted(seen,key=lambda item:(item not in chosen_ids,seen[item][0]))
                            seen={item:seen[item] for item in ranked[:64]}
                        if remember_results:
                            remember_segment(origin,current,states)
                        work.results=_refill_results(seen.values(),max_edges)
                    break
        return work.results
    recipe_limit=continuation.__dict__.setdefault("recipe_limit",
        max(0,max_expansions-stats.get("expansions",0)) if max_expansions is not None else None)
    return continuation.advance(key,recipes,deadline=deadline,stats=stats,
        max_expansions=max_expansions,recipe_limit=recipe_limit)
