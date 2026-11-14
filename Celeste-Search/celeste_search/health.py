"""Read-only live progress diagnostics. Alerts never reset or pause the game."""
def inspect_progress(journal,map_data=None):
    frame=journal.get("frame",0)
    observation=journal.get("last_observation") or {}
    exploration=journal.get("exploration",{})
    events=journal.get("events",[])
    errors=[]
    recent_no_route=[]
    for event in reversed(events):
        if event.get("event") in {"segment_completed","room_entered","berry_confirmed","walls_opened"}:
            break
        if event.get("event")=="no_offline_route":
            recent_no_route.append(event)
    if len(recent_no_route)>=3:
        errors.append({"kind":"search_stalled","failed_batches":len(recent_no_route),
                       "room":observation.get("room")})
    progress_frames=max(0,frame-exploration.get("last_progress_frame",frame))
    repeated={edge:count for edge,count in exploration.get("stalled_edges",{}).items() if count>=2}
    if repeated and (progress_frames>=600 or "last_progress_frame" not in exploration):
        errors.append({"kind":"equivalent_state_room_cycle","edges":repeated})
    if progress_frames>=1800:
        errors.append({"kind":"no_new_observed_area","control_frames":progress_frames})
    if journal.get("telemetry_error"):
        errors.append({"kind":"telemetry_error","message":journal["telemetry_error"]})
    pending=[]
    if map_data:
        from .planner import ordinary_berries,berry_id
        confirmed=set(journal.get("confirmed_berries",[]))
        for name in journal.get("visited_rooms",[]):
            for entity in ordinary_berries(map_data.rooms[name]):
                identity=berry_id(name,entity)
                if identity not in confirmed:
                    pending.append(identity)
    return {"frame":frame,"room":observation.get("room"),"phase":journal.get("phase"),
            "destination":exploration.get("active_destination"),
            "entered_rooms":len(journal.get("visited_rooms",[])),
            "confirmed_berries":journal.get("confirmed_berries",[]),
            "opened_walls":exploration.get("opened_walls",[]),
            "pending_berries_in_entered_rooms":pending,
            "pending_does_not_mean_proven_reachable":True,"alerts":errors}
