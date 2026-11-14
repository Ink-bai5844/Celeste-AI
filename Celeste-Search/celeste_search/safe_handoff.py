"""Search-only checks for handing a cassette-room route back to the player."""
from functools import lru_cache

from .physics import JUMP, LEFT, RIGHT


@lru_cache(maxsize=1024)
def neutral_handoff_safe(simulator, state, frames=120):
    """Predict the neutral tail without consuming any executable commands."""
    if state.dead:
        return False
    current = state
    for _ in range(frames):
        current = simulator.step(current, 0)
        if current.dead or current.room != state.room:
            return False
    return True


def target_approach_macros(state, goal):
    """Walk, brake, then jump vertically near a selected target.

    Geometry supplies only a relative timing estimate. Search checks every
    input and rejects an unsafe first target hit, including pickup contact.
    """
    if (not getattr(goal, "handoff_frames", 0) or goal.kind != "target"
            or state.dead or not state.on_ground or state.vy < 0 or state.room != goal.room):
        return
    distance = goal.x-state.x
    rise = state.y-goal.y
    if abs(distance) > 160 or not 4 < rise < 64:
        return
    direction = RIGHT if distance >= 0 else LEFT
    estimate = round(abs(distance)/90*60)
    seen = set()
    for offset in (0, -2, 2, -4, 4, -6, 6):
        walk = max(0, min(110, estimate+offset))
        for quiet in (0, 2, 4, 8):
            release = (0,) if walk+quiet == 0 and state.prev_input & JUMP else ()
            actions = (direction,)*walk + (0,)*quiet + release + (JUMP,)*40
            if actions not in seen:
                seen.add(actions)
                yield actions, len(actions)
