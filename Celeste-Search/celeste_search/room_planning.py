"""Bounded offline room planning for the running, unpaused controller.

All candidates are replayed from observed state. Collection is a separate
checkpoint, and complete computed traces preserve momentum through room
transitions. A room crossing alone is not a reason to release every input.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from .maps import load_map
from .physics import SimState, Simulator
from .planner import (Goal, Heuristic, berry_id, coverage_goals, ordinary_berries,
                      resume_search_alternatives, room_is_supported, secret_goals, winged_berries)
from .routes import snapshot
from .strategies import search_goal


_CONTEXTS = {}


def _context(path):
    """Keep immutable geometry and simulator indexes in persistent workers."""
    path=Path(path)
    if not path.exists():  # Synthetic/test maps supplied through load_map.
        data=load_map(path)
        return data,Simulator(data)
    stat=path.stat()
    key=(str(path.resolve()),stat.st_mtime_ns,stat.st_size)
    if key not in _CONTEXTS:
        if len(_CONTEXTS)>=3:
            _CONTEXTS.clear()
        data=load_map(path)
        _CONTEXTS[key]=(data,Simulator(data))
    return _CONTEXTS[key]


def action_fingerprint(actions: list[int]) -> str:
    return sha256(bytes(actions)).hexdigest()


def _trace_to_boundary(simulator, start, actions, *, stop_at_first=True):
    state, used, states = start, [], [start]
    for mask in actions:
        successor = simulator.step(state, mask)
        if successor.dead:
            break
        state = successor
        used.append(mask)
        states.append(state)
        if stop_at_first and state.room != start.room:
            break
    return used, state, states


def _safe_anchor(simulator, state):
    return (not state.dead and state.on_ground and state.state == 0
            and abs(state.vx) < .01 and abs(state.vy) < .01
            and not state.following and simulator._safe_ground(state)
            and not state.dash_cooldown and not state.refill_cooldown)


def _finish_safely(simulator, start, actions, *, allow_transitions=False):
    """Keep only a continuously simulated route with a safe idle endpoint.

    A near-goal airborne state is not useful if releasing controls kills the
    player while the next background job runs. Try its real coast, then earlier
    support states along the same path. No manufactured landing is permitted.
    """
    used, final, states = _trace_to_boundary(simulator, start, actions,stop_at_first=not allow_transitions)
    if not allow_transitions and final.room != start.room:
        return used, final, states
    indices = [len(used)] + [i for i in range(len(used)-1, 0, -1)
                            if states[i].on_ground and i % 6 == 0]
    for index in indices[:32]:
        state = states[index]
        extension = []
        for _ in range(150):
            if _safe_anchor(simulator, state):
                trial = simulator.roll(state, 0, 12)
                if (_safe_anchor(simulator, trial) and trial.room == state.room
                        and abs(trial.x-state.x) < 1 and abs(trial.y-state.y) < 1):
                    return used[:index]+[0]*len(extension), state, states[:index+1]+extension
            state = simulator.step(state, 0)
            if state.dead or (not allow_transitions and state.room != start.room):
                break
            extension.append(state)
    return [], start, [start]


def _neighbor_routes(map_data, simulator, start, visited, exploration):
    """Order first hops by directed distance to outstanding map work.

    At a cleared leaf the next destination is a frontier elsewhere in the graph,
    rather than the nearest visited room. Traversal counts break ties; failed
    goals are deferred, never silently recorded as completed.
    """
    supported = {name for name, room in map_data.rooms.items()
                 if room.spawns and room_is_supported(simulator, name)}
    graph = {name: [r.name for r in map_data.neighbors(name) if r.name in supported]
             for name in supported}
    failures = exploration.get("failed_goals", {})
    cells={tuple(cell) for cell in exploration.get("visited_cells",[])}
    outstanding = {}
    for name in supported:
        room = map_data.rooms[name]
        berry_goals = [Goal("berry",name,e.x,e.y,berry_id(name,e))
                      for e in ordinary_berries(room) if berry_id(name,e) not in start.berries]
        if name not in visited:
            outstanding[name] = (0,0.0)
        elif berry_goals and name != start.room:
            outstanding[name] = (1,min(min(8,failures.get(g.key,0))*2 for g in berry_goals))
        elif name != start.room and any(
                not g.achieved(start) and (name,int(g.x//16),int(g.y//16)) not in cells
                for g in secret_goals(room)):
            outstanding[name]=(2,0)
        elif name != start.room and cells:
            surfaces=[g for g in coverage_goals(room)
                      if (name,int(g.x//16),int(g.y//16)) not in cells]
            if surfaces:
                outstanding[name]=(3,min(min(8,failures.get(g.key,0))*2 for g in surfaces))
    counts = exploration.get("edge_visits", {})
    visits = exploration.get("room_visits", {})
    deferred=exploration.get("destination_deferred",{})
    recent=exploration.get("recent_edges",[])
    stalled=exploration.get("stalled_edges",{})
    # Select one map destination before searching any first hop. Independent
    # nearest-frontier choices at successive room entries cause A-B-A cycles.
    active=exploration.get("active_destination")
    if active not in supported or (active==start.room and _safe_anchor(simulator,start)):
        active=None
    if active:
        destination=active
    else:
        queue,distances=deque([start.room]),{start.room:0}
        while queue:
            name=queue.popleft()
            for other in graph.get(name,[]):
                if other not in distances:
                    distances[other]=distances[name]+1
                    queue.append(other)
        options=[(category,distances[name]+cost+max(0,deferred.get(name,0))*100,name)
                 for name,(category,cost) in outstanding.items()
                 if name!=start.room and name in distances]
        destination=min(options)[-1] if options else None
    ranked = []
    for next_name in graph.get(start.room, []):
        queue, distances = deque([next_name]), {next_name: 1}
        while queue:
            name = queue.popleft()
            for other in graph[name]:
                # A first hop that only reaches work by immediately returning
                # through this room is a dead end, not an alternate route.
                if (other != start.room or destination==start.room) and other not in distances:
                    distances[other] = distances[name]+1
                    queue.append(other)
        cost=distances.get(destination,math.inf)
        category=0 if math.isfinite(cost) else math.inf
        room = map_data.rooms[next_name]
        goal = Goal("room",next_name,*room.spawns[0])
        edge=f"{start.room}>{next_name}"
        reverse=f"{next_name}>{start.room}"
        # Recrosses remain possible for momentum/transition jumps. Only known
        # equivalent-state stalls get a substantial penalty; recency breaks ties.
        rank = (category,cost+min(6,failures.get(goal.key,0))*2+min(12,stalled.get(edge,0))*3,
                int(bool(recent and recent[-1]==reverse)),
                counts.get(f"{start.room}>{next_name}",0), visits.get(next_name,0), next_name)
        ranked.append((rank, room, destination or next_name))
    return sorted(ranked, key=lambda row: row[0])


def plan_room(map_path: str, start_state: dict, seconds: float = 10,
              expansions: int = 3000, variant: int = 0,
              forbidden: list[str] | None = None,
              visited_rooms: list[str] | None = None,
              exploration: dict | None = None) -> dict:
    """Return an offline candidate and explicit unfinished exploration work.

    ``exploration`` contains actual visited_cells (room,x//16,y//16), room_visits,
    edge_visits, failed_goals and failed_regions from the controller. Generated
    geometry is never counted as explored. ``primary_goal`` identifies the
    attempted target even for an intermediate safe waypoint.
    """
    if not math.isfinite(seconds) or seconds <= 0 or expansions < 1:
        raise ValueError("seconds and expansions must be positive finite budgets")
    map_data,simulator = _context(map_path)
    start = SimState.from_dict(start_state)
    if start.room not in map_data.rooms:
        raise ValueError(f"Unknown start room {start.room!r}")
    exploration = exploration or {}
    blacklist = set(forbidden or [])
    if blacklist:
        resume_search_alternatives(simulator,start)
    visited = set(visited_rooms or []) | set(exploration.get("room_visits",{})) | {start.room}
    visited_cells = {tuple(cell) for cell in exploration.get("visited_cells",[])}
    failures = exploration.get("failed_goals",{})
    started = time.monotonic()
    deadline = started+seconds
    total_expansions = total_simulated = 0
    candidates, attempted = [], []
    rejected = 0
    room = map_data.rooms[start.room]
    winged={berry_id(room.name,e) for e in winged_berries(room)}-start.berries

    def empty(status, reason):
        return dict(actions=[],expected=[snapshot(start)],final_state=start.to_dict(),
                    goals=[],primary_goal=attempted[0]["goal"] if attempted else None,
                    status=status,reason=reason,fingerprint=action_fingerprint([]),
                    expansions=total_expansions,simulated_frames=total_simulated,
                    can_execute=False,attempts=attempted,runtime_verified=False,
                    destination_room=exploration.get("active_destination"),
                    navigation_revision=exploration.get("navigation_revision",0))

    if start.dead:
        return empty("blocked","start_state_dead")
    if not room_is_supported(simulator,start.room):
        result=empty("unsupported","unsupported_current_room")
        result["capabilities"]=simulator.capability_report(start.room)
        return result
    fled=sorted(winged & start.fled)
    if fled:
        result=empty("retry","winged_berry_fled_current_room_respawn_needed")
        result.update(primary_goal=fled[0],pending_goals=fled,requires_room_respawn=True)
        return result

    def record(actions, goal, reason, priority,destination=None):
        nonlocal total_simulated,rejected
        used,final,states=_finish_safely(simulator,start,actions,allow_transitions=True)
        total_simulated+=len(used)
        fingerprint=action_fingerprint(used)
        if fingerprint in blacklist:
            rejected+=1
            return None
        berries=len(final.berries-start.berries)
        transition=final.room!=start.room
        traversed=any(goal.achieved(state) for state in states) if goal.kind in {"room","hidden"} else False
        if transition and ((goal.kind == "room" and not traversed)
                           or (goal.kind != "room" and not berries and not goal.achieved(final))):
            return None
        displacement=math.hypot(final.x-start.x,final.y-start.y)
        cells={(s.room,int(s.x//16),int(s.y//16)) for s in states}
        new_cells=len(cells-visited_cells)
        if not used or not (transition or berries or displacement>=16 or final.broken!=start.broken):
            return None
        if not transition and not berries and visited_cells and not goal.achieved(final):
            heuristic=Heuristic(map_data,goal)
            if heuristic(final)>heuristic(start)-8 and new_cells<3:
                return None
        region_penalty=sum(1 for region in exploration.get("failed_regions",[])
                           if region.get("room")==final.room
                           and math.hypot(region.get("x",0)-final.x,region.get("y",0)-final.y)<24)
        achieved=goal.achieved(final) or traversed
        boundaries=[i for i in range(1,len(states)) if states[i-1].room!=states[i].room]
        candidate=dict(actions=used,expected=[snapshot(s) for s in states],
                       final_state=json.loads(json.dumps(final.to_dict())),
                       goals=[dict(goal=goal.key,kind=goal.kind,room=goal.room,reached=achieved)],
                       primary_goal=goal.key,status="transition" if transition else "partial",
                       reason="continuous_room_route" if boundaries else reason,
                       safe_endpoint=_safe_anchor(simulator,final),fingerprint=fingerprint,can_execute=True,
                       boundary_indices=boundaries,allow_room_continuation=bool(boundaries),
                       destination_room=destination or exploration.get("active_destination"),
                       navigation_revision=exploration.get("navigation_revision",0),
                       predicted_new_cells=new_cells,runtime_verified=False)
        rank=(berries,priority,int(transition),
              int(achieved), -region_penalty, new_cells,
              min(displacement,160),-len(used))
        candidates.append((rank,candidate))
        return candidate

    def work(goal, budget_end, priority,destination=None):
        nonlocal total_expansions,total_simulated
        remaining=expansions-total_expansions
        if remaining<=0 or time.monotonic()>=min(deadline,budget_end):
            return None
        if winged and not goal.no_dash_until:
            goal=replace(goal,no_dash_until=sorted(winged)[0])
        end=min(deadline,budget_end)
        search_end=time.monotonic()+(end-time.monotonic())*.7 if goal.kind=="room" else end
        result=search_goal(simulator,start,goal,deadline=search_end,
                           max_expansions=max(1,remaining*3//4) if goal.kind=="room" else remaining,
                           max_frames=900,variant=variant,
                           allow_unsupported=False)
        total_expansions+=result.expansions
        total_simulated+=result.simulated_frames
        attempted.append(dict(goal=goal.key,kind=goal.kind,reached=result.reached,reason=result.reason))
        actions=list(result.actions)
        current=result.state
        if current.room!=start.room:
            # Keep computed momentum across boundaries. A first-hop room is a
            # milestone, not an instruction to release input mid-transition.
            targets=[]
            if destination and current.room!=destination:
                target_room=map_data.rooms[destination]
                targets.append(Goal("room",destination,*target_room.spawns[0]))
            supports=coverage_goals(map_data.rooms[current.room])
            supports.sort(key=lambda g:(math.hypot(current.x-g.x,current.y-g.y),g.key))
            targets+=supports[:3]
            for target in targets:
                safe_actions,safe_final,_=_finish_safely(simulator,start,actions,allow_transitions=True)
                if (safe_final.room==current.room and _safe_anchor(simulator,safe_final)
                        and (not destination or current.room==destination)):
                    actions=safe_actions
                    break
                remaining=expansions-total_expansions
                if remaining<1 or time.monotonic()>=end or len(actions)>=900:
                    break
                tail=search_goal(simulator,current,target,deadline=end,
                                 max_expansions=remaining,max_frames=900-len(actions),
                                 variant=variant,allow_unsupported=False)
                total_expansions+=tail.expansions
                total_simulated+=tail.simulated_frames
                attempted.append(dict(goal=target.key,kind=target.kind,reached=tail.reached,
                                      reason=tail.reason,continuation=True))
                if tail.actions:
                    actions+=tail.actions
                    current=tail.state
                if tail.reached and target.kind=="room":
                    supports=coverage_goals(map_data.rooms[current.room])
                    supports.sort(key=lambda g:(math.hypot(current.x-g.x,current.y-g.y),g.key))
                    targets[1:]=supports[:3]
        return record(actions,goal,"goal_checkpoint" if result.reached else "safe_waypoint",priority,destination)

    berries=[Goal("berry",room.name,e.x,e.y,berry_id(room.name,e)) for e in ordinary_berries(room)
             if berry_id(room.name,e) not in start.berries]
    berries.sort(key=lambda g:(g.id not in winged,failures.get(g.key,0),
                               math.hypot(g.x-start.x,g.y-start.y),g.key))
    for goal in berries[:2]:
        share=.72 if failures.get(goal.key,0)<3 else .35
        candidate=work(goal,started+seconds*share,4)
        if candidate and (candidate["goals"][0]["reached"] or candidate["status"]=="transition"):
            break
    checkpoint=any(c[0][0]>0 for c in candidates)
    secrets=[g for g in secret_goals(room) if not g.achieved(start)
             and (g.kind=="break" or (g.room,int(g.x//16),int(g.y//16)) not in visited_cells)]
    secrets.sort(key=lambda g:(failures.get(g.key,0),math.hypot(g.x-start.x,g.y-start.y),g.key))
    if not checkpoint and not winged:
        for goal in secrets[:2]:
            candidate=work(goal,started+seconds*.86,4)
            if candidate and candidate["goals"][0]["reached"]:
                checkpoint=True
                break
    if not checkpoint:
        neighbors=_neighbor_routes(map_data,simulator,start,visited,exploration)
        neighbors=[row for row in neighbors if math.isfinite(row[0][0])]
        for index,(_,other,destination) in enumerate(neighbors):
            remaining=max(0,deadline-time.monotonic())
            share=1 if index==len(neighbors)-1 else .8
            goal=Goal("room",other.name,*other.spawns[0])
            candidate=work(goal,time.monotonic()+remaining*share,3 if index==0 else 1,destination)
            if candidate:
                candidate["destination_room"]=destination
            if candidate and candidate["status"]=="transition":
                break

    if not any(c[1]["status"]=="transition" or c[0][0]>0 for c in candidates):
        surfaces=[g for g in coverage_goals(room)
                  if 24<=math.hypot(start.x-g.x,start.y-g.y)<=220
                  and (g.room,int(g.x//16),int(g.y//16)) not in visited_cells]
        surfaces.sort(key=lambda g:(failures.get(g.key,0),math.hypot(start.x-g.x,start.y-g.y),g.key))
        for goal in surfaces[:2]:
            work(goal,deadline,1)
    if not candidates:
        return empty("retry" if rejected else "no_progress",
                     "forbidden_candidates" if rejected else "no_meaningful_safe_route")
    selected=max(candidates,key=lambda row:row[0])[1]
    selected.update(expansions=total_expansions,simulated_frames=total_simulated,
                    attempts=attempted,wall_seconds=round(time.monotonic()-started,3))
    return selected
