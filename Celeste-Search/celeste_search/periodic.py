"""Match a repeating spring phase while the real game keeps running."""

def can_seek_spring_recovery(observation,map_data):
    if (observation.get("dead") or observation.get("state")!=0
            or observation.get("on_ground") or observation.get("in_cutscene")
            or abs(observation.get("vx",999))>.1 or observation.get("vy",0)>=-150):
        return False
    room=map_data.rooms.get(observation.get("room"))
    if room is None:
        return False
    return any(e.kind=="spring" and abs(e.x-observation["x"])<=16
               and 0<=e.y-observation["y"]<=160 for e in room.entities)


def periodic_anchor_matches(observation,anchor):
    if (observation.get("dead") or observation.get("state")!=0
            or observation.get("room")!=anchor.get("room")
            or observation.get("spawn_id")!=anchor.get("spawn_id")
            or observation.get("dashes")!=anchor.get("dashes")):
        return False
    if any(abs(observation[k]-anchor[k])>1.01 for k in ("x","y","vx","vy")):
        return False
    if any(set(observation.get(k,[]))!=set(anchor.get(k,[])) for k in ("berries","following")):
        return False
    current,initial=observation.get("model_state",{}),anchor.get("model_state",{})
    for key in ("var_jump","auto_jump_active","dash_cooldown","refill_cooldown",
                "jump_buffer","dash_buffer","force_frames","facing","ducking"):
        if current.get(key)!=initial.get(key):
            return False
    if current.get("prev_input",0)&112 or initial.get("prev_input",0)&112:
        return False
    # A spring resets its launch position/velocity, but Actor can retain a
    # subpixel remainder across cycles. A <=1px positional difference is
    # explicit and remains subject to the normal execution drift guard.
    return True


def projected_anchor_matches(observation,state,origin):
    """Accept the predicted future input boundary, including subpixel phase."""
    expected=dict(origin)
    expected.update({k:state[k] for k in ("room","x","y","vx","vy","dashes","berries","following")})
    expected["model_state"]=state
    if not periodic_anchor_matches(observation,expected):
        return False
    if observation.get("state")!=state.get("state") or observation.get("on_ground")!=state.get("on_ground"):
        return False
    if any(abs(observation[k]-state[k])>.01 for k in ("x","y","vx","vy")):
        return False
    if abs(observation.get("stamina",-999)-state.get("stamina",0))>.05:
        return False
    model=observation.get("model_state",{})
    if not all(abs(model.get(k,999)-state[k])<.0001 for k in ("rem_x","rem_y")):
        return False
    for key in ("var_speed","auto_jump","grace","force_x","force_frames",
                "wall_retention","wall_retained","hop_wait_x","hop_wait_x_speed",
                "lift_timer","lift_vx","lift_vy","max_fall","dash_attack"):
        if key in state and abs(model.get(key,999)-state[key])>.01:
            return False
    return True
