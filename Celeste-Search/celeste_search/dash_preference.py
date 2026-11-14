"""Resource preference for already validated routes; never an admission gate."""
from __future__ import annotations

from .physics import DASH


def dash_usage(candidate):
    """Use dash-start events from the uninterrupted physical validation replay.

    Charge drops alone miss a refill and a dash in the same update. The
    conservative input/charge fallback serves older/minimal callers only.
    Movement progress excludes stationary waiting, so delaying a button by
    padding a route with neutral frames cannot improve its resource preference.
    """
    actions = candidate.get("actions", ())
    states = candidate.get("expected", ())
    complete = len(states) == len(actions) + 1
    charges = complete and all(type(state.get("dashes")) is int for state in states)
    frames = []
    pressed = []
    previous_mask = candidate.get("anchor_state", {}).get("prev_input", 0)
    for index, action in enumerate(actions, 1):
        if charges:
            used = states[index]["dashes"] < states[index-1]["dashes"]
        else:
            used = bool(action & DASH and not previous_mask & DASH)
        if used:
            frames.append(index)
        if action & DASH and not previous_mask & DASH:
            pressed.append(index)
        previous_mask = action
    if "dash_start_frames" in candidate:
        events = candidate["dash_start_frames"]
        if (not isinstance(events,list) or any(type(index) is not int
                or not 1<=index<=len(actions) for index in events)
                or events!=sorted(set(events))):
            raise ValueError("invalid replay dash-start events")
        frames = list(events)
    elif len(pressed)>len(frames):
        frames = pressed
    elif not frames and candidate.get("anchor_state",{}).get("dash_buffer",0)>0 and actions:
        frames = [1]  # Legacy traces cannot prove an inherited buffer unused.
    progress = [0]
    for index in range(1, len(actions)+1):
        moved = (complete and any(states[index].get(key) != states[index-1].get(key)
                                  for key in ("room", "x", "y")))
        progress.append(progress[-1] + bool(moved))
    return dict(dash_count=len(frames), dash_start_frames=frames,
                movement_before_dashes=[progress[index-1] for index in frames],
                movement_frames=progress[-1], action_frames=len(actions))


def dash_preference_key(candidate):
    usage = dash_usage(candidate)
    total = max(1, usage["movement_frames"])
    return (usage["dash_count"],
            tuple(-count/total for count in usage["movement_before_dashes"]),
            usage["action_frames"])
