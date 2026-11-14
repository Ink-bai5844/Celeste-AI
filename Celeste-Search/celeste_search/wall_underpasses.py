"""Release actual static walls and descend below overhangs using real inputs."""
from __future__ import annotations

from dataclasses import replace
import time

from .physics import DASH, DOWN, GRAB, LEFT, RIGHT, UP
from .search_progress import remember_segment
from .wall_transfers import _static_contact


def _recipes(origin, target, allow_dash):
    waypoint = getattr(target, "waypoint", None)
    x, _ = waypoint(origin) if waypoint is not None else (target.x, target.y)
    toward = RIGHT if x >= origin.x else LEFT
    face = (RIGHT if target.facing > 0 else LEFT) if target.kind == "wall_grip" else toward
    catches = (GRAB | UP | face, GRAB | face)
    waits = (32, 28, 24, 20, 16, 12, 8, 4)
    # DOWN suppresses slow wall sliding and requests ordinary fast fall.
    # Releasing horizontal input is a separate useful way to clear a ledge.
    for wait in waits:
        for release in (DOWN | toward, DOWN):
            for catch in catches:
                yield ((release,) * wait + (catch,) * 140,
                       f"drop_catch:{wait}:{release}:{catch}")
    if not allow_dash or origin.dashes <= 0:
        return
    for wait in waits:
        for release in (DOWN | toward, DOWN):
            for aim in (toward, UP | toward, DOWN | toward):
                for catch in catches:
                    yield ((release,) * wait + (DASH | aim,) * 6 + (catch,) * 140,
                           f"drop_dash:{wait}:{release}:{aim}:{catch}")


def wall_underpass_edges(simulator, origin, target, *, deadline,
                         max_frames=220, stats=None, max_expansions=None,
                         max_edges=4, continuation=None, remember_results=True):
    """Return bounded ``(actions, state, label)`` drop-and-catch transfers.

    Real static contact is mandatory. The caller's simulator and strict target
    predicate remain authoritative for every frame, including public limits,
    moving contacts, hazards, dash buffers and any room crossing.
    """
    if continuation is not None:
        return _continued_underpasses(simulator, origin, target, deadline=deadline,
            max_frames=max_frames, stats=stats, max_expansions=max_expansions,
            max_edges=max_edges, continuation=continuation,
            remember_results=remember_results)
    if max_frames <= 0 or max_edges <= 0 or time.monotonic() >= deadline:
        return []
    stats = stats if stats is not None else {}
    limit = min(max_expansions if max_expansions is not None else float("inf"),
                stats.get("max_expansions", float("inf")),
                stats.get("expansions", 0) + 128)
    if stats.get("expansions", 0) >= limit or _static_contact(simulator, origin) is None:
        return []
    if target.achieved(origin):
        return [([], origin, "wall_underpass:already_reached:dash_starts=0")]
    allow_dash = not getattr(simulator, "no_dash", False)
    seen, refined_seen, successful, recipes_seen = {}, {}, [], set()
    initial_expansions = stats.get("expansions", 0)
    refinement_quota = min(8, int(limit-initial_expansions)//4)

    def roll(recipe, label, *, refined=False):
        if time.monotonic() >= deadline or stats.get("expansions", 0) >= limit:
            return
        recipe = recipe[:max_frames]
        if recipe in recipes_seen:
            return
        recipes_seen.add(recipe)
        stats["expansions"] = stats.get("expansions", 0) + 1
        state, actions, states, dash_starts = origin, [], [origin], []
        for mask in recipe:
            if time.monotonic() >= deadline:
                break
            state = simulator.step(state, mask)
            started_dash = bool(getattr(simulator, "last_step_started_dash", False))
            if started_dash:
                dash_starts.append(len(actions) + 1)
            stats["simulated_frames"] = stats.get("simulated_frames", 0) + 1
            actions.append(mask)
            states.append(state)
            if state.dead or not allow_dash and (mask & DASH or started_dash):
                break
            if target.achieved(state):
                key = replace(state, frame=0, reason="")
                rank = (len(dash_starts), -state.dashes, -state.stamina,
                        len(actions), tuple(-tick for tick in dash_starts))
                row = (rank, actions, state, label, states)
                bucket = refined_seen if refined else seen
                if key not in bucket or rank < bucket[key][0]:
                    bucket[key] = row
                if not refined:
                    successful.append((row, recipe))
                return
            if state.room != origin.room:
                break

    for recipe, label in _recipes(origin, target, allow_dash):
        # Reserve refinement work only after a real contact exists. Failed
        # searches retain the original full recipe budget.
        base_limit = limit-refinement_quota if successful else limit
        if time.monotonic() >= deadline or stats.get("expansions", 0) >= base_limit:
            break
        roll(recipe, label)

    refined_attempts = 0
    for row, recipe in sorted(successful, key=lambda item:item[0][0]):
        wait = 1
        while wait < len(recipe) and recipe[wait] == recipe[0]:
            wait += 1
        if wait == len(recipe):
            continue
        for offset in (-1, 1):
            if (refined_attempts >= refinement_quota or time.monotonic() >= deadline
                    or stats.get("expansions", 0) >= limit):
                break
            # Change only the release duration of a successful baseline;
            # dash aim and catch remain intact. Never refine another variant.
            varied = (recipe[0],)*(wait+offset) + recipe[wait:]
            if wait+offset <= 0 or varied[:max_frames] in recipes_seen:
                continue
            refined_attempts += 1
            roll(varied, row[3]+f":refined:{offset:+d}", refined=True)

    result = []
    for rank, actions, state, label, states in _underpass_rows(seen, refined_seen, max_edges):
        if remember_results:
            remember_segment(origin, state, states)
        family = "wall_underpass_refined" if ":refined:" in label else "wall_underpass"
        result.append((actions, state, f"{family}:{label}:dash_starts={rank[0]}"))
    return result


def _underpass_rows(seen, refined_seen, max_edges):
    originals = sorted(seen.values(), key=lambda row:row[0])
    refinements = sorted((row for key,row in refined_seen.items() if key not in seen),
                         key=lambda row:row[0])
    # Preserve the best original and one genuinely distinct timing neighbor,
    # including when a coarse arrival band would otherwise merge them.
    chosen = originals[:1]
    if max_edges > 1 and refinements:
        chosen.append(refinements[0])
    deferred, bands = [], set()
    def band(row):
        rank, _, state, _, _ = row
        return (rank[0], state.dashes, state.state, round(state.y/8),
                round(state.vx/45), round(state.vy/45), int(state.stamina/20))
    bands.update(band(row) for row in chosen)
    for row in sorted(originals+refinements, key=lambda row:row[0]):
        if len(chosen) >= max_edges:
            break
        if any(row is existing for existing in chosen):
            continue
        if band(row) in bands:
            deferred.append(row)
        else:
            chosen.append(row)
            bands.add(band(row))
    chosen.extend(deferred[:max(0,max_edges-len(chosen))])
    return sorted(chosen, key=lambda row:row[0])


def _continued_underpasses(simulator, origin, target, *, deadline, max_frames,
                           stats, max_expansions, max_edges, continuation,
                           remember_results):
    from .macro_continuation import MACRO_EXPANSION
    if stats is None:
        stats = continuation.__dict__.setdefault("stats", {})
    key = (id(simulator), id(stats), origin, target, max_frames, max_edges,
           remember_results)
    limit = min(max_expansions if max_expansions is not None else float("inf"),
                stats.get("max_expansions", float("inf")))
    recipe_limit = continuation.__dict__.setdefault("recipe_limit",
        min(128, max(0, limit - stats.get("expansions", 0))))

    def recipes(work):
        if (max_frames <= 0 or max_edges <= 0
                or _static_contact(simulator, origin) is None):
            return []
        if target.achieved(origin):
            return [([], origin, "wall_underpass:already_reached:dash_starts=0")]
        allow_dash = not getattr(simulator, "no_dash", False)
        seen, refined_seen, successful, recipes_seen = {}, {}, [], set()
        refinement_quota = min(8, int(recipe_limit) // 4)

        def roll(recipe, label, *, refined=False):
            recipe = recipe[:max_frames]
            if recipe in recipes_seen:
                return
            recipes_seen.add(recipe)
            yield MACRO_EXPANSION
            state, actions, states, dash_starts = origin, [], [origin], []
            for mask in recipe:
                state = simulator.step(state, mask)
                started_dash = bool(getattr(simulator, "last_step_started_dash", False))
                if started_dash:
                    dash_starts.append(len(actions) + 1)
                stats["simulated_frames"] = stats.get("simulated_frames", 0) + 1
                actions.append(mask)
                states.append(state)
                if state.dead or not allow_dash and (mask & DASH or started_dash):
                    break
                if target.achieved(state):
                    identity = replace(state, frame=0, reason="")
                    rank = (len(dash_starts), -state.dashes, -state.stamina,
                            len(actions), tuple(-tick for tick in dash_starts))
                    row = (rank, actions, state, label, ())
                    bucket = refined_seen if refined else seen
                    if identity not in bucket or rank < bucket[identity][0]:
                        bucket[identity] = row
                        if remember_results:
                            remember_segment(origin, state, states)
                    if not refined:
                        successful.append((row, recipe))
                    work.results = [(masks, end,
                        f"{'wall_underpass_refined' if ':refined:' in name else 'wall_underpass'}"
                        f":{name}:dash_starts={order[0]}")
                        for order, masks, end, name, _ in
                        _underpass_rows(seen, refined_seen, max_edges)]
                    return
                if state.room != origin.room:
                    break
                if len(actions) % 16 == 0:
                    yield None

        for recipe, label in _recipes(origin, target, allow_dash):
            base_limit = recipe_limit - refinement_quota if successful else recipe_limit
            if work.expansions >= base_limit:
                break
            yield from roll(recipe, label)

        refined_attempts = 0
        for row, recipe in sorted(successful, key=lambda item:item[0][0]):
            wait = 1
            while wait < len(recipe) and recipe[wait] == recipe[0]:
                wait += 1
            if wait == len(recipe):
                continue
            for offset in (-1, 1):
                if refined_attempts >= refinement_quota or work.expansions >= recipe_limit:
                    break
                varied = (recipe[0],) * (wait + offset) + recipe[wait:]
                if wait + offset <= 0 or varied[:max_frames] in recipes_seen:
                    continue
                refined_attempts += 1
                yield from roll(varied, row[3] + f":refined:{offset:+d}", refined=True)
        return work.results

    return continuation.advance(key, recipes, deadline=deadline, stats=stats,
        max_expansions=limit, recipe_limit=recipe_limit)
