"""Named offline search recipes and a full-state graph of physical supports."""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
import time

from .planner import Goal, Heuristic, SearchResult, coverage_goals, search_leg, state_key
from .physics import LEFT,RIGHT,_truth
from .strategies import RefillGoal, SpringGoal, WallSpringGoal, UnderhangGoal, usable_spring, spring_goal
from .search_progress import remember_segment,search_tracker
from .geometry_milestones import CassetteSupportGoal, cassette_support_goals


_SPECS = (
    ("ordered_mechanism","ordered","连续机关接力",4,0),
    ("ordered_mechanism_capability","ordered","机关接力备选",4,0),
    ("ordered_corridor","ordered","多通路轮换验证",0,0),
    ("ordered_regional","ordered","区域起跳与精确收尾",0,0),
    ("rested_high_refill","rested_support","高台恢复再补充",4,0),
    ("support_beam_climb","support_graph","多级攀升",4,0),
    ("wall_relay","wall_relay","攀墙换台再出发",4,0),
    ("spring_relay","spring_relay","弹簧借力接力",1,0),
    ("underhang_relay","underhang_relay","落边折返下绕",2,0),
    ("zip_chain_graph","zip_graph","移动平台接力",4,0),
    ("direct_greedy","direct","快速直达",0,0),
    ("semantic_refill","semantic","空中补充",2,0),
    ("rested_high_direct","rested_support","高台恢复直达",4,0),
    ("support_beam_forward","support_graph","多级前进",0,0),
    ("direct_fine","direct","精细动作",4,0),
    ("zip_board_ride","zip_graph","搭乘移动平台",0,0),
    ("zip_chain_fine","zip_graph","移动平台接力精调",4,0),
    ("zip_chain_semantic","zip_graph","移动平台接力落台",5,0),
    ("semantic_landing_near","semantic","近侧落台",1,0),
    ("rested_alternate_refill","rested_support","备选高台",4,1),
    ("support_beam_underhang","support_graph","下绕通道",2,0),
    ("direct_balanced","direct","平衡直达",1,0),
    ("semantic_landing_far","semantic","远侧落台",4,0),
    ("support_beam_refill","support_graph","补充接力",5,0),
    ("rested_third_refill","rested_support","第三高台",4,2),
    ("direct_long","direct","长动作跨越",3,0),
    ("ordered_fine","ordered","顺序精细搜索",4,0),
    ("ordered_balanced","ordered","顺序均衡搜索",1,0),
    ("ordered_greedy","ordered","顺序快速搜索",0,0),
    ("ordered_long","ordered","顺序长动作",3,0),
    ("ordered_short_jump","ordered","顺序短跳",2,0),
    ("ordered_precision","ordered","精确状态分支",4,0),
    ("ordered_micro","ordered","单帧微调",4,0),
    ("ordered_lookahead","ordered","兼顾下一路线点",0,0),
    ("ordered_phase_diverse","ordered","保留前段备选",0,0),
    ("ordered_wide","ordered","宽搜索前沿",3,0),
    ("ordered_refill_wait","ordered","等待补充恢复",0,0),
    ("ordered_semantic","ordered","自动补充与折返",0,0),
)


def route_strategy_specs(route=None, state=None, map_data=None):
    """Cheap deterministic portfolio metadata; no map loading or game calls.

    Multi-point/constrained routes retain ordered monitoring, including the
    corridor family which backtracks across public waypoint arrivals. Other
    semantic profiles remain independent alternatives.
    """
    restricted=False
    if route is not None:
        points=route.get("waypoints",[route])
        restricted=len(points)!=1 or any(any(segment.values()) for segment in route.get("segments",[]))
    result=[]
    for name,family,label,variant,index in _SPECS:
        if restricted and family!="ordered":
            continue
        if name in {"ordered_mechanism","ordered_mechanism_capability"} and map_data is not None:
            room_name=state.get("room") if isinstance(state,dict) else getattr(state,"room",None)
            room=map_data.rooms.get(room_name) if map_data is not None else None
            broken=state.get("broken",()) if isinstance(state,dict) else getattr(state,"broken",())
            can_break=(route is None or any(not segment.get("no_dash")
                       for segment in (route.get("segments") or [{}])))
            # Opening a passage is a mechanism action even in a room without
            # movers. This inventory decision grants no resources or damage;
            # the ordered solver must prove the break and the whole suffix.
            if room is None or not any((e.kind=="zipMover" and e.nodes) or
                    (can_break and e.kind=="dashBlock" and _truth(e.attrs.get("canDash",True))
                     and f"{room.name}:{e.id}" not in broken) for e in room.entities):
                continue
        if map_data is not None and state is not None and family=="zip_graph":
            room_name=state.get("room") if isinstance(state,dict) else state.room
            room=map_data.rooms.get(room_name)
            if room is None or not any(entity.kind=="zipMover" for entity in room.entities):
                continue
        if map_data is not None and state is not None and family=="spring_relay":
            room_name=state.get("room") if isinstance(state,dict) else state.room
            room=map_data.rooms.get(room_name)
            if room is None or not any(usable_spring(entity) for entity in room.entities):
                continue
        result.append(dict(name=name,family=family,label=label,description=label,
                           variant=variant,support_index=index,
                           key_precision="exact" if name=="ordered_precision" else "coarse",
                           ordered_profile=name.removeprefix("ordered_") if family=="ordered" else "default"))
        if name in {"ordered_regional", "ordered_corridor"}:
            # It monitors ordered constraints, but explores a different source
            # frontier. Give it a fair slot beside the other algorithm families.
            result[-1]["portfolio_family"]=name.removeprefix("ordered_")
        if name in {"ordered_corridor","ordered_regional","ordered_mechanism","ordered_mechanism_capability"}:
            result[-1]["budget_scale"]=2.5
            # A corridor expansion may include long physical macro rollouts.
            # Give its backtracking time without increasing its node budget
            # or the controller's overall request deadline.
            result[-1]["time_budget_scale"]=11.25
            result[-1]["max_seconds"]=90
    clock=(state.get("cassette_clock") if isinstance(state,dict) else
           getattr(state,"cassette_clock",None))
    if (route is not None and len(route.get("waypoints",[route]))>1 and clock is not None
            and not any(any(segment.values()) for segment in route.get("segments",[]))):
        # The continuous platform recipe is the useful first proposal here;
        # direct ordered variants still remain available as alternatives.
        result.sort(key=lambda spec:spec["name"]!="ordered_semantic")
        for spec in result:
            if spec["name"]=="ordered_semantic":
                spec["budget_scale"]=2.5
    elif (route is not None and len(route.get("waypoints",[route]))>1
          and not any(any(segment.values()) for segment in route.get("segments",[]))):
        from .underhang_moves import route_has_underhang
        if route_has_underhang(route,state,map_data):
            result.sort(key=lambda spec:spec["name"]!="ordered_semantic")
    if (route is not None and len(route.get("waypoints",[route]))>1
            and not any(any(segment.values()) for segment in route.get("segments",[]))
            and map_data is not None and state is not None):
        # Reuse the exact internal-guidance decision. A nearby refill alone
        # does not establish a useful relay on this ordered public route.
        from .assist_planning import _refill_guidance_route
        from .physics import SimState
        actual=SimState.from_dict(state) if isinstance(state,dict) else state
        guidance_route=dict(route,segments=route.get("segments") or
                            [{} for _ in route["waypoints"]])
        guidance,_=_refill_guidance_route(map_data,actual,guidance_route)
        if len(guidance["waypoints"])>len(route["waypoints"]):
            for spec in result:
                if spec["name"]=="ordered_semantic":
                    spec["portfolio_family"]="refill_relay"
            result.sort(key=lambda spec:spec["name"]!="ordered_semantic")
    return result


def strategy_spec(name):
    for spec in route_strategy_specs():
        if spec["name"]==name:
            return spec
    raise ValueError(f"Unknown route search strategy: {name}")


@dataclass(frozen=True)
class ZipBoardGoal(Goal):
    identity: str=""
    width: float=8

    def location(self,state):
        record=next((row for row in state.mechanisms if row[0]==self.identity),None)
        return (record[3],record[4]) if record else (self.x-self.width/2,self.y)

    def waypoint(self,state):
        x,y=self.location(state)
        return x+self.width/2,y

    def achieved(self,state):
        if state.dead or state.room!=self.room or state.vy<0:
            return False
        x,y=self.location(state)
        return (state.on_ground and abs(state.y-round(y))<.1
                and round(x)+4<=state.x<=round(x)+self.width-4)


@dataclass(frozen=True)
class TransientSupportGoal(Goal):
    """Board a currently present crumble/falling support at its actual position."""
    identity: str=""
    width: float=8
    mechanism_kind: str="crumbleBlock"

    def location(self,state):
        record=next((row for row in state.mechanisms if row[0]==self.identity),None)
        if self.identity in state.broken or (record and record[1]==
                (3 if self.mechanism_kind=="crumbleBlock" else 4)):
            return None
        return (round(record[3]),round(record[4])) if record else (self.x-self.width/2,self.y)

    def waypoint(self,state):
        location=self.location(state)
        return (location[0]+self.width/2,location[1]) if location else (self.x,self.y)

    def achieved(self,state):
        location=self.location(state)
        return bool(location and not state.dead and state.room==self.room
                    and state.on_ground and state.vy>=0 and abs(state.y-location[1])<.1
                    and location[0]+4<=state.x<=location[0]+self.width-4)


def _transient_supports(room,state):
    for entity in room.entities:
        if entity.kind not in {"crumbleBlock","fallingBlock"} or entity.width<8:
            continue
        identity=f"{room.name}:{entity.id}"
        goal=TransientSupportGoal("surface",room.name,entity.x+entity.width/2,entity.y,
                 id="support:"+identity,identity=identity,width=entity.width,mechanism_kind=entity.kind)
        if goal.location(state) is not None:
            yield goal


def _zip_ride_edges(simulator,state,goal,max_frames,*,deadline=math.inf,stats=None):
    """Retain real launch states before a later ceiling crush or failed ride.

    A zip's terminal position need not leave enough headroom for a rider. The
    earlier rising states are independent physical edges, not failed attempts
    that should disappear when a later neutral frame would be fatal.
    """
    current=state
    actions=[]
    states=[state]
    preview=search_tracker(state)
    for tick in range(1,min(100,max_frames)+1):
        if time.monotonic()>=deadline:
            return
        successor=simulator.step(current,0)
        if stats is not None:
            stats["simulated_frames"]=stats.get("simulated_frames",0)+1
        if successor.dead or successor.room!=state.room or not goal.achieved(successor):
            return
        current=successor
        actions.append(0)
        states.append(current)
        record=next((row for row in current.mechanisms if row[0]==goal.identity),None)
        complete=bool(record and record[1]==3)
        if preview:
            preview.report(lambda:[("alternative",current,states)],0,tick)
        if tick in (8,16,24,32) or complete:
            remember_segment(state,current,states)
            yield list(actions),current,"ride" if complete else f"launch:{tick}"
        if complete:
            return


def _ride_zip(simulator,state,goal,max_frames):
    """Legacy complete ride; graph/relay callers also keep partial launches."""
    for actions,arrival,label in _zip_ride_edges(simulator,state,goal,max_frames):
        if label=="ride":
            return actions,arrival
    return None


def _support_candidates(simulator,state,goal,mode,visited):
    from .zip_grips import zip_top_blocked,zip_side_grip_goals
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    room=data.rooms[state.room]
    heuristic=Heuristic(data,goal)
    current_h=heuristic(state)
    candidates=[]
    supports=coverage_goals(room)
    supports=list(supports)+cassette_support_goals(room,state)
    if mode in {"climb","underhang"}:
        supports=list(supports)+list(_transient_supports(room,state))
    for support in supports:
        rise=state.y-support.y
        distance=math.hypot(state.x-support.x,rise)
        if not 16<distance<400 or rise>240:
            continue
        if support.achieved(state) or support.key in visited:
            continue
        if state.cassette_clock is not None:
            # A permanently occupied standing hitbox cannot be a landing;
            # the exposed side can still be an independent wall-grip target.
            if simulator.solid_at(state.room,support.x,support.y,state=state,dynamic_rects=()):
                continue
        probe=replace(state,x=support.x,y=support.y)
        progress=current_h-heuristic(probe)
        score=distance-progress*.7
        if mode=="climb":
            score-=max(-40,rise)*.8
        elif mode=="underhang":
            score-=max(0,-rise)*.6
        if state.cassette_clock is not None:
            # A high final ledge can geometrically dominate every lower beat
            # block while being unreachable with the current dash. Keep short
            # physical refill hops ahead of repeated attempts at that ledge.
            score+=max(0,rise-48)*2.5
            if rise>96:
                score+=240
        # Wide-ranging graph legs must not require an intermediate support to
        # immediately reduce distance. Backward climbs stay in the beam.
        # Graph edges retain arrival momentum; a separate neutral branch below
        # can restore resources. Requiring a perfectly rested centre on every
        # narrow ledge would discard useful moving-platform launches.
        candidates.append((score,support.key,support))
    for entity in room.entities:
        identity=f"{room.name}:{entity.id}"
        distance=math.hypot(state.x-entity.x,state.y-entity.y)
        if distance>420:
            continue
        if entity.kind=="zipMover" and entity.nodes:
            if zip_top_blocked(entity,room):
                continue
            key="zip:"+identity
            if key not in visited:
                target=ZipBoardGoal("zip_board",room.name,entity.x+entity.width/2,entity.y,
                                    id=key,identity=identity,width=entity.width)
                score=distance-(180 if mode in {"zip","zip_chain"} else 70)
                candidates.append((score,key,target))
        elif entity.kind=="refill" and identity not in dict(state.refill_timers):
            key="refill:"+identity
            if key not in visited:
                score=distance-(150 if mode=="refill" else 30)
                candidates.append((score,key,RefillGoal("refill",room.name,entity.x,entity.y+5,id=key)))
        elif usable_spring(entity):
            key="spring:"+identity
            if key not in visited:
                candidates.append((distance-60,key,spring_goal(room.name,entity)))
        elif mode=="underhang" and entity.kind=="jumpThru":
            key="under:"+identity
            if key not in visited:
                candidates.append((distance-80,key,UnderhangGoal("passage",room.name,
                                  entity.x+entity.width/2,entity.y+24,id=key)))
    if mode in {"climb","underhang"}:
        from .wall_moves import wall_grip_goals
        grips=wall_grip_goals(simulator,state,goal)
        selected=grips[:3]
        if state.cassette_clock is not None:
            # Dynamic sides must not crowd out the permanent ledge edges that
            # let a player leave a sequence of disappearing supports.
            static=[grip for grip in grips if not grip.identity and grip.y < state.y-16]
            static.sort(key=lambda grip:math.hypot(grip.x-goal.x,grip.y-goal.y)
                        +math.hypot(grip.x-state.x,grip.y-state.y)*.4)
            selected+=static[:2]
        for grip in selected:
            if grip.key not in visited:
                distance=math.hypot(state.x-grip.x,state.y-grip.y)
                bonus=100 if state.cassette_clock is not None and not grip.identity else 80
                standing_on_face=(mode=="climb" and state.on_ground and goal.y<state.y-16
                    and any(block.id==grip.identity and block.collidable
                            and abs(state.y-block.y)<.1
                            and state.x+4>block.x and state.x-4<block.x+block.width
                            for block in state.cassette_blocks))
                # Dropping to grab the side of one's current support rarely
                # advances an ascent. Keep the branch, behind higher ledges.
                candidates.append((distance-bonus+(120 if standing_on_face else 0),grip.key,grip))
    blocked={f"{room.name}:{entity.id}" for entity in room.entities
             if entity.kind=="zipMover" and zip_top_blocked(entity,room)}
    for grip in zip_side_grip_goals(simulator,state,goal) if blocked or state.cassette_clock is not None else ():
        already_present=any(getattr(row[2],"identity","")==grip.identity
                            and getattr(row[2],"facing",None)==grip.facing for row in candidates)
        if (grip.identity in blocked or state.cassette_clock is not None) and grip.key not in visited and not already_present:
            distance=math.hypot(state.x-grip.x,state.y-grip.y)
            candidates.append((distance-80,grip.key,grip))
    return sorted(candidates,key=lambda row:row[:2])


def cassette_wait_edges(simulator, start, *, deadline, max_frames=160,stats=None):
    """Keep full-state launch phases reached by neutral input on real support.

    Waiting is an ordinary simulated edge, never a clock edit. A disappearing
    support or any loss of ground stops this waiting branch before execution.
    """
    if start.cassette_clock is None or not start.on_ground:
        return
    state=start
    actions=[]
    states=[start]
    emitted=set()
    previous_signature=(state.cassette_clock.current_index,state.cassette_clock.beat_index)
    for _ in range(min(max_frames,180)):
        if time.monotonic()>=deadline:
            return
        successor=simulator.step(state,0)
        if stats is not None:
            stats['simulated_frames']=stats.get('simulated_frames',0)+1
        if successor.dead or successor.room!=start.room or not successor.on_ground:
            return
        state=successor
        actions.append(0)
        states.append(state)
        clock=state.cassette_clock
        signature=(clock.current_index,clock.beat_index)
        # Keep a pre-toggle and a new-color launch, not every beat. Repeated
        # nearly identical phases otherwise consume the support graph's beam.
        if (abs(state.vx)<.01 and state.dashes>0 and signature!=previous_signature
                and clock.beat_index%8 in {0,7} and signature not in emitted):
            emitted.add(signature)
            remember_segment(start,state,states)
            yield list(actions),state,f"beat:{clock.beat_index}"
        previous_signature=signature


def search_support_graph(simulator,start,goal,*,deadline,max_expansions,max_frames,
                         mode="forward",variant=0,beam_width=8,max_depth=6):
    """Bounded beam over alternative *simulated* intermediate arrival states.

    Each edge is a search_leg replay from its full predecessor state. Search can
    revisit a different branch after a failed suffix, including stationary and
    moving supports. It never edits positions, velocity, resources or mechanisms.
    """
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    if start.cassette_clock is not None:
        max_depth=max(max_depth,10)
    heuristic=Heuristic(data,goal)
    nodes=[(start,[],(),0)]
    frontier=[(heuristic(start),0)]
    seen={state_key(start)}
    expanded=simulated=0
    best=SearchResult([],start,False,0,0,"support_graph_budget")
    best_h=heuristic(start)
    attempts=[]
    while frontier and time.monotonic()<deadline and expanded<max_expansions:
        _,index=heapq.heappop(frontier)
        state,prefix,visited,depth=nodes[index]
        if len(prefix)>=max_frames:
            continue
        # Probe the final destination from every actual support arrival. Reserve
        # time to branch rather than letting the first unsuccessful suffix win.
        remaining=deadline-time.monotonic()
        leg_budget=min(.6 if state.cassette_clock is not None else 1.4,max(.12,remaining*.2))
        # The final point can be far above a lower relay platform. Spending a
        # full direct probe at every phase is unnecessary before reaching an
        # actual climbing/landing state near that elevation.
        if state.cassette_clock is not None and state.on_ground and state.y-goal.y>104:
            leg_budget=min(leg_budget,.08)
        direct=search_leg(simulator,state,goal,deadline=min(deadline,time.monotonic()+leg_budget),
                          max_expansions=min(1600,max_expansions-expanded),
                          max_frames=max_frames-len(prefix),variant=variant,allow_unsupported=False)
        expanded+=direct.expansions
        simulated+=direct.simulated_frames
        if direct.reached:
            answer=SearchResult(prefix+direct.actions,direct.state,True,expanded,simulated,"support_graph_goal")
            answer.strategy_details=dict(internal_stages=list(visited),nodes=len(nodes),legs=attempts)
            return answer
        h=heuristic(direct.state)
        if not direct.state.dead and h<best_h:
            best=SearchResult(prefix+direct.actions,direct.state,False,expanded,simulated,"support_graph_budget")
            best_h=h
        if depth>=max_depth:
            continue
        if state.cassette_clock is not None and state.on_ground and not any(item.startswith("wait:") for item in visited[-1:]):
            wait_stats={}
            for wait_actions,arrival,label in cassette_wait_edges(simulator,state,deadline=deadline,
                    max_frames=min(160,max_frames-len(prefix)),stats=wait_stats):
                arrival_key=state_key(arrival)
                if arrival_key in seen:
                    continue
                seen.add(arrival_key)
                child=len(nodes)
                nodes.append((arrival,prefix+wait_actions,visited+("wait:"+label,),depth))
                heapq.heappush(frontier,(heuristic(arrival)+len(prefix+wait_actions)*.03+100-depth*20,child))
            simulated+=wait_stats.get('simulated_frames',0)
        options=_support_candidates(simulator,state,goal,mode,visited)
        if mode=="zip" and not visited:
            options=[row for row in options if isinstance(row[2],ZipBoardGoal)]+[row for row in options if not isinstance(row[2],ZipBoardGoal)]
        chosen=options[:4]
        dynamic=next((row for row in options if isinstance(row[2],(ZipBoardGoal,RefillGoal,SpringGoal,CassetteSupportGoal))
                      or row[2].kind=="wall_grip"),None)
        if dynamic is not None and dynamic not in chosen:
            chosen=chosen[:3]+[dynamic]
        cassette=next((row for row in options if isinstance(row[2],CassetteSupportGoal)),None)
        if cassette is not None and cassette not in chosen:
            chosen=chosen[:3]+[cassette]
        if state.cassette_clock is not None:
            ledge=next((row for row in options if row[2].kind=='wall_grip'
                        and not getattr(row[2],'identity','') and row[2].y<state.y-16),None)
            if ledge is not None and ledge not in chosen:
                reserved=[row for row in (cassette,ledge) if row is not None]
                chosen=[row for row in chosen if row not in reserved][:4-len(reserved)]+reserved
        spring=next((row for row in options if isinstance(row[2],(SpringGoal,WallSpringGoal))),None)
        if spring is not None and spring not in chosen:
            # Keep an independent real spring-trigger branch even when a closer
            # wall, refill or ledge would otherwise occupy every graph slot.
            reserved=[row for row in (dynamic,spring) if row is not None]
            chosen=[row for row in chosen if row not in reserved][:4-len(reserved)]+reserved
        for _,key,target in chosen:
            if time.monotonic()>=deadline or expanded>=max_expansions:
                break
            remaining=deadline-time.monotonic()
            leg_budget=min(.8 if state.cassette_clock is not None else 1.8,max(.15,remaining*.22))
            physical=[]
            if isinstance(target,CassetteSupportGoal):
                from .cassette_edges import cassette_support_edges
                macro_stats={'max_expansions':max(0,max_expansions-expanded)}
                physical=list(cassette_support_edges(simulator,state,target,
                    # The earliest arrivals can all be short no-dash hops to
                    # one side. Keep enough samples for an earlier dash landing
                    # at the other side, with its distinct continuation window.
                    deadline=min(deadline,time.monotonic()+min(.6,leg_budget)),
                    max_frames=max_frames-len(prefix),max_edges=6,stats=macro_stats))
                simulated+=macro_stats.get('simulated_frames',0)
                expanded+=macro_stats.get('expansions',0)
            if physical:
                first_actions,first_state,_=physical[0]
                edge=SearchResult(first_actions,first_state,True,0,0,"cassette_support_macro")
            else:
                edge=search_leg(simulator,state,target,deadline=min(deadline,time.monotonic()+leg_budget),
                                max_expansions=min(1600,max_expansions-expanded),
                                max_frames=max_frames-len(prefix),variant=4 if target.kind=="surface" else variant,
                                allow_unsupported=False)
            expanded+=edge.expansions
            simulated+=edge.simulated_frames
            attempts.append(dict(goal=key,reached=edge.reached,frames=len(edge.actions),
                                 depth=depth,start_x=state.x,start_y=state.y,
                                 beat=state.cassette_clock.beat_index if state.cassette_clock else None))
            if not edge.reached or edge.state.dead:
                continue
            choices=[(actions,arrival,key+":"+label) for actions,arrival,label in physical] or [(edge.actions,edge.state,key)]
            if isinstance(target,CassetteSupportGoal):
                from .cassette_edges import cassette_launch_edges
                launch_stats={}
                for approach,landed,_ in list(choices):
                    for launch,arrival,label in cassette_launch_edges(simulator,landed,target,
                            deadline=deadline,max_frames=min(40,max_frames-len(prefix)-len(approach)),
                            stats=launch_stats):
                        choices.append((approach+launch,arrival,key+":"+label))
                simulated+=launch_stats.get('simulated_frames',0)
                for approach,landed,label in list(choices):
                    settled=landed
                    neutral=[]
                    states=[settled]
                    for _ in range(min(12,max_frames-len(prefix)-len(approach))):
                        successor=simulator.step(settled,0)
                        simulated+=1
                        if not target.achieved(successor):
                            break
                        settled=successor
                        neutral.append(0)
                        states.append(settled)
                        if abs(settled.vx)<.01 and settled.dashes>0:
                            remember_segment(landed,settled,states)
                            choices.append((approach+neutral,settled,label+":restored"))
                            break
            if target.kind=="wall_grip":
                if getattr(target,"mechanism_kind","")=="zipMover":
                    from .zip_grips import zip_grip_edges
                    grip_stats=dict(max_expansions=max(0,max_expansions-expanded))
                    launches=zip_grip_edges(simulator,edge.state,target,goal,deadline=deadline,
                               max_frames=min(140,max_frames-len(prefix)-len(edge.actions)),stats=grip_stats)
                else:
                    from .wall_moves import wall_launch_edges
                    grip_stats=None
                    launches=wall_launch_edges(simulator,edge.state,
                               max_frames=min(140,max_frames-len(prefix)-len(edge.actions)))
                for inputs,arrival,label in launches:
                    if grip_stats is None:
                        simulated+=len(inputs)
                    choices.insert(0,(edge.actions+inputs,arrival,key+":"+label))
                if grip_stats is not None:
                    simulated+=grip_stats.get("simulated_frames",0)
                    expanded+=grip_stats.get("expansions",0)
            elif isinstance(target,ZipBoardGoal):
                ride_stats={}
                for ride_actions,arrival,label in _zip_ride_edges(simulator,edge.state,target,
                        max_frames-len(prefix)-len(edge.actions),deadline=deadline,stats=ride_stats):
                    choices.insert(0,(edge.actions+ride_actions,arrival,key+":"+label))
                simulated+=ride_stats.get("simulated_frames",0)
            elif target.kind=="surface" and edge.state.on_ground and edge.state.vy>=0:
                settled=edge.state
                neutral=[]
                neutral_states=[settled]
                for _ in range(min(40,max_frames-len(prefix)-len(edge.actions))):
                    successor=simulator.step(settled,0)
                    simulated+=1
                    if successor.dead or successor.room!=state.room or not successor.on_ground:
                        break
                    settled=successor
                    neutral.append(0)
                    neutral_states.append(settled)
                    if (abs(settled.vx)<.01 and abs(settled.vy)<.01
                            and settled.dashes>=getattr(simulator,"max_dashes",1)):
                        remember_segment(edge.state,settled,neutral_states)
                        choices.append((edge.actions+neutral,settled,key+":settled"))
                        break
            for actions,arrival,label in choices:
                arrival_key=state_key(arrival)
                if arrival_key in seen:
                    continue
                seen.add(arrival_key)
                child=len(nodes)
                nodes.append((arrival,prefix+actions,visited+(key,),depth+1))
                score=heuristic(arrival)+len(prefix+actions)*.03
                if start.cassette_clock is not None:
                    # Give a newly reached physical relay a chance before
                    # revisiting the same entrance at another waiting phase.
                    score-=min(depth+1,5)*20
                    score+=(35 if arrival.dashes==0 else 0)+abs(arrival.vx)*.04
                if mode=="climb":
                    score+=max(0,arrival.y-goal.y)*.25
                if mode in {"zip","zip_chain"} and label.endswith(":ride"):
                    score-=50
                heapq.heappush(frontier,(score,child))
        if len(frontier)>beam_width:
            frontier=heapq.nsmallest(beam_width,frontier)
            heapq.heapify(frontier)
    best.expansions=expanded
    best.simulated_frames=simulated
    best.strategy_details=dict(nodes=len(nodes),legs=attempts)
    return best


def zip_opens_target(entity,room,state,goal):
    """Whether activating this mover vacates a selected boundary aperture."""
    if (entity.kind!="zipMover" or not entity.nodes
            or not getattr(goal,"transition_to","")
            or (getattr(goal,"source_room","") or goal.room)!=room.name):
        return False
    record=next((row for row in state.mechanisms if row[0]==f"{room.name}:{entity.id}"),None)
    x,y=(record[3],record[4]) if record else (entity.x,entity.y)
    tx,ty=entity.nodes[0]
    px,py=goal.tile_x+4,goal.tile_y+4
    edge=goal.edge
    if edge in {"left","right"}:
        overlap=y<=py<y+entity.height
        near=min(abs(x-px),abs(x+entity.width-px))<=64
        clears=(tx>=x+8 if edge=="left" else tx<=x-8)
    else:
        overlap=x<=px<x+entity.width
        near=min(abs(y-py),abs(y+entity.height-py))<=64
        clears=(ty>=y+8 if edge=="top" else ty<=y-8)
    return bool(overlap and near and clears)


def search_zip_grip_relay(simulator,start,goal,*,deadline,max_expansions,max_frames,variant=0,
                           identities=None,opening=False):
    """Use real side contacts to ride a spiked mover or open a blocked passage."""
    from .zip_grips import zip_side_grip_goals,zip_grip_edges,zip_top_blocked
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    heuristic=Heuristic(data,goal)
    expanded=simulated=0;stages=[]
    best=SearchResult([],start,False,0,0,"zip_grip_budget")
    best_h=heuristic(start)
    if not opening and best_h<144 and time.monotonic()<deadline:
        # A player already released beside the destination should continue
        # directly, rather than being forced back onto the mover they just rode.
        remaining=deadline-time.monotonic()
        direct=search_leg(simulator,start,goal,
                  deadline=min(deadline,time.monotonic()+min(3.,remaining*.55)),
                  max_expansions=min(3000,max_expansions),max_frames=max_frames,
                  variant=variant,allow_unsupported=False)
        expanded+=direct.expansions;simulated+=direct.simulated_frames
        if direct.reached:
            direct.strategy_details=dict(internal_stages=[],side_grip=True)
            return direct
        if not direct.state.dead and heuristic(direct.state)<best_h:
            best,best_h=direct,heuristic(direct.state)
    blocked=(set(identities) if identities is not None else
             {f"{start.room}:{entity.id}" for entity in data.rooms[start.room].entities
              if entity.kind=="zipMover" and zip_top_blocked(entity,data.rooms[start.room])})
    grips=[grip for grip in zip_side_grip_goals(simulator,start,goal) if grip.identity in blocked]
    if opening:
        # The destination-facing face can be inside the closed aperture.
        # Activate from an accessible near face, retaining the opposite face
        # as a later alternative rather than demanding it before the door moves.
        grips.sort(key=lambda grip:(math.hypot(grip.x-start.x,grip.y-start.y),grip.key))
    for grip in grips[:4]:
        remaining=deadline-time.monotonic()
        if remaining<=0 or expanded>=max_expansions:
            break
        approach=search_leg(simulator,start,grip,
                 deadline=min(deadline,time.monotonic()+min(3.,remaining*.4)),
                 max_expansions=min(2500,max_expansions-expanded),max_frames=max_frames,
                 variant=0,allow_unsupported=False)
        expanded+=approach.expansions;simulated+=approach.simulated_frames
        stages.append(dict(goal=grip.key,reached=approach.reached,frames=len(approach.actions)))
        if not approach.reached:
            continue
        ride_stats=dict(expansions=0,simulated_frames=0,max_expansions=max_expansions-expanded)
        rides=list(zip_grip_edges(simulator,approach.state,grip,goal,deadline=deadline,
                   max_frames=min(140,max_frames-len(approach.actions)),stats=ride_stats))
        expanded+=ride_stats["expansions"];simulated+=ride_stats["simulated_frames"]
        choices=[(approach.actions,approach.state,"grip")]
        choices.extend((approach.actions+inputs,arrival,label) for inputs,arrival,label in rides)
        if opening:
            drop_stats=dict(expansions=0,simulated_frames=0,max_expansions=max_expansions-expanded)
            for inputs,arrival,label in _coast_support_arrivals(simulator,approach.state,goal,
                    deadline=deadline,max_frames=max_frames-len(approach.actions),stats=drop_stats):
                # A real landing precedes native dash refill by an update. Let
                # that update happen physically while the actuator keeps moving.
                if len(approach.actions)+len(inputs)<max_frames and time.monotonic()<deadline:
                    settled=simulator.step(arrival,0);drop_stats["simulated_frames"]+=1
                    if not settled.dead and settled.room==arrival.room and settled.on_ground and settled.vy>=0:
                        remember_segment(arrival,settled,[arrival,settled])
                        inputs=inputs+[0];arrival=settled
                choices.append((approach.actions+inputs,arrival,"opening_drop:"+label))
            expanded+=drop_stats["expansions"];simulated+=drop_stats["simulated_frames"]
        choices.sort(key=lambda entry:(0 if opening and entry[2].startswith("opening_drop:")
                                         and entry[1].on_ground and entry[1].dashes else 1,
                             heuristic(entry[1])+len(entry[0])*.02,len(entry[0])))
        for index,(prefix,arrival,label) in enumerate(choices[:4]):
            remaining=deadline-time.monotonic()
            if remaining<=0 or expanded>=max_expansions or len(prefix)>=max_frames:
                break
            if goal.achieved(arrival):
                suffix=SearchResult([],arrival,True,0,0,"zip_grip_goal")
            else:
                share=remaining*.65 if index==0 and len(choices)>1 else remaining*.5
                suffix=search_leg(simulator,arrival,goal,
                       deadline=min(deadline,time.monotonic()+share),
                       max_expansions=max(1,max_expansions-expanded),
                       max_frames=max_frames-len(prefix),variant=0,allow_unsupported=False)
            expanded+=suffix.expansions;simulated+=suffix.simulated_frames
            stages.append(dict(goal=label,reached=suffix.reached,frames=len(prefix)+len(suffix.actions)))
            if suffix.reached:
                answer=SearchResult(prefix+suffix.actions,suffix.state,True,expanded,simulated,"zip_grip_goal")
                answer.strategy_details=dict(internal_stages=stages,side_grip=True)
                return answer
            if not suffix.state.dead and heuristic(suffix.state)<best_h:
                best_h=heuristic(suffix.state)
                best=SearchResult(prefix+suffix.actions,suffix.state,False,0,0,"zip_grip_budget")
    best.expansions=expanded;best.simulated_frames=simulated
    best.strategy_details=dict(internal_stages=stages,side_grip=True)
    return best


def search_zip_chain(simulator,start,goal,*,deadline,max_expansions,max_frames,variant=0,ride_frames=None):
    """A resource-efficient zip relay recipe leaves most budget for its suffix.

    The separate graph recipe explores alternative boarding states and supports;
    this one follows nearby useful moving platforms, retaining every real ride
    frame and its lift velocity, then spends the remaining search on the target.
    """
    from .strategies import search_goal
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    from .zip_grips import zip_top_blocked
    zips=[entity for entity in data.rooms[start.room].entities if entity.kind=="zipMover" and entity.nodes]
    opening={f"{start.room}:{entity.id}" for entity in zips
             if zip_opens_target(entity,data.rooms[start.room],start,goal)}
    if opening:
        return search_zip_grip_relay(simulator,start,goal,deadline=deadline,
                 max_expansions=max_expansions,max_frames=max_frames,variant=variant,
                 identities=opening,opening=True)
    if zips and all(zip_top_blocked(entity,data.rooms[start.room]) for entity in zips):
        return search_zip_grip_relay(simulator,start,goal,deadline=deadline,
                 max_expansions=max_expansions,max_frames=max_frames,variant=variant)
    heuristic=Heuristic(data,goal)
    state=start
    actions=[]
    visited=set()
    expanded=simulated=0
    stages=[]

    def launch_goal(board,label):
        nonlocal expanded,simulated
        if (goal.y>=state.y-48 or abs(goal.x-state.x)>160
                or len(actions)>=max_frames or expanded>=max_expansions
                or time.monotonic()>=deadline):
            return None
        from .zip_launches import zip_launch_edges
        stats=dict(expansions=0,simulated_frames=0,max_expansions=max_expansions-expanded)
        launch_deadline=min(deadline,time.monotonic()+min(2.,max(0,deadline-time.monotonic())*.35))
        choices=[]
        for inputs,arrival,kind in zip_launch_edges(simulator,state,board,goal,
                  deadline=launch_deadline,max_frames=min(160,max_frames-len(actions)),stats=stats):
            if goal.achieved(arrival):
                choices.append((actions+inputs,arrival,kind))
            if len(choices)>=3:
                break
        expanded+=stats["expansions"];simulated+=stats["simulated_frames"]
        stages.append(dict(goal=label+":launch",reached=bool(choices),alternatives=len(choices)))
        if not choices:
            return None
        inputs,arrival,kind=choices[0]
        result=SearchResult(inputs,arrival,True,expanded,simulated,"zip_launch_goal")
        result.strategy_details=dict(internal_stages=list(stages),launch=kind)
        # The ordered frontier admits every alternative through its shared
        # frame monitor. A failed return can reopen a different real first-hit
        # velocity/resource state instead of committing this first arrival.
        result.alternative_actions=[choice[0] for choice in choices[1:]]
        return result

    for _ in range(3):
        options=[]
        for entity in data.rooms[state.room].entities:
            key=f"{state.room}:{entity.id}"
            if entity.kind!="zipMover" or not entity.nodes or key in visited:
                continue
            if zip_top_blocked(entity,data.rooms[state.room]):
                continue
            tx,ty=entity.nodes[0]
            destination=replace(state,x=tx+entity.width/2,y=ty)
            progress=heuristic(state)-heuristic(destination)
            record=next((row for row in state.mechanisms if row[0]==key),None)
            current_x,current_y=(record[3],record[4]) if record else (entity.x,entity.y)
            distance=math.hypot(state.x-(current_x+entity.width/2),state.y-current_y)
            if progress>=-24 and distance<320:
                options.append((distance-progress*.15,key,entity))
        if not options or deadline<=time.monotonic() or expanded>=max_expansions:
            break
        _,key,entity=min(options,key=lambda row:row[:2])
        visited.add(key)
        board=ZipBoardGoal("zip_board",state.room,entity.x+entity.width/2,entity.y,
                            id="zip:"+key,identity=key,width=entity.width)
        now=time.monotonic()
        board_budget=(min(4.,(deadline-now)*.6) if ride_frames is not None
                      else min(2.5,(deadline-now)*.22))
        edge=search_leg(simulator,state,board,deadline=min(deadline,now+board_budget),
                        max_expansions=min(2000,max_expansions-expanded),
                        max_frames=max_frames-len(actions),variant=variant%6,allow_unsupported=False)
        expanded+=edge.expansions
        simulated+=edge.simulated_frames
        if not edge.reached:
            stages.append(dict(goal=key,reached=False))
            continue
        actions+=edge.actions
        state=edge.state
        stages.append(dict(goal=key,reached=True,frames=len(edge.actions)))
        board_stage=stages[-1]
        launched=launch_goal(board,key+":board")
        if launched is not None:
            return launched
        ride_stats={}
        rides=list(_zip_ride_edges(simulator,state,board,max_frames-len(actions),
                                  deadline=deadline,stats=ride_stats))
        simulated+=ride_stats.get("simulated_frames",0)
        full=next((ride for ride in rides if ride[2]=="ride"),None)
        ride=(full if ride_frames is None and full else
              min(rides,key=lambda edge:(abs(len(edge[0])-(ride_frames or 24)),-len(edge[0])))
              if rides else None)
        if ride is not None:
            ride_actions,state,ride_label=ride
            actions+=ride_actions
            board_stage["ride_frames"]=len(ride_actions)
            board_stage["ride_kind"]=ride_label
            launched=launch_goal(board,key+":"+ride_label)
            if launched is not None:
                return launched
    if time.monotonic()>=deadline or expanded>=max_expansions or len(actions)>=max_frames:
        result=SearchResult(actions,state,False,expanded,simulated,"zip_chain_budget")
    else:
        finder=search_goal if variant==5 else search_leg
        suffix=finder(simulator,state,goal,deadline=deadline,max_expansions=max_expansions-expanded,
                       max_frames=max_frames-len(actions),variant=variant,allow_unsupported=False)
        result=SearchResult(actions+suffix.actions,suffix.state,suffix.reached,
                            expanded+suffix.expansions,simulated+suffix.simulated_frames,suffix.reason)
    result.strategy_details=dict(internal_stages=stages)
    return result


def search_wall_relay(simulator,start,goal,*,deadline,max_expansions,max_frames,variant=4):
    """Search a real wall grip, reverse onto a support, then continue the goal.

    Some higher launch ledges can only be reached by first approaching a wall
    farther forward. Those temporarily backward landings remain alternatives;
    no position, velocity, resource or mechanism state is edited between stages.
    """
    from .wall_moves import wall_grip_goals,wall_launch_edges
    from .strategies import search_goal
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    heuristic=Heuristic(data,goal)
    expanded=simulated=0;stages=[]
    best=SearchResult([],start,False,0,0,"wall_relay_budget")
    best_h=heuristic(start)
    grips=[]
    for grip in wall_grip_goals(simulator,start,goal)[:3]:
        extra_height=min(16,max(0,grip.grip_offset-12))
        if extra_height:
            grips.append(replace(grip,y=grip.y-extra_height,
                         grip_offset=grip.grip_offset-extra_height,id=grip.id+":higher"))
        grips.append(grip)
    for grip_index,grip in enumerate(grips):
        remaining=deadline-time.monotonic()
        if remaining<=0 or expanded>=max_expansions:
            break
        approach=search_leg(simulator,start,grip,
                    deadline=min(deadline,time.monotonic()+min(3.,remaining*.35)),
                    max_expansions=min(2500,max_expansions-expanded),max_frames=max_frames,
                    variant=variant,allow_unsupported=False)
        expanded+=approach.expansions;simulated+=approach.simulated_frames
        stages.append(dict(goal=grip.key,reached=approach.reached,frames=len(approach.actions),
                           arrival=dict(x=approach.state.x,y=approach.state.y,
                                        stamina=approach.state.stamina,state=approach.state.state)))
        if not approach.reached:
            continue
        candidates=[]
        for actions,arrival,label in wall_launch_edges(simulator,approach.state,
                                            max_frames=min(140,max_frames-len(approach.actions))):
            simulated+=len(actions)
            if arrival.dead or not arrival.on_ground or arrival.vy<0 or not arrival.dashes:
                continue
            prefix=approach.actions+actions
            if len(prefix)>=max_frames:
                continue
            candidates.append((heuristic(arrival)+len(prefix)*.03,label,prefix,arrival))
        candidates.sort(key=lambda entry:entry[:2])
        chosen=candidates[:4]
        if candidates and goal.y<start.y:
            highest=min(candidates,key=lambda entry:(entry[3].y,entry[0]))
            if highest not in chosen:
                chosen=chosen[:3]+[highest]
        suffix_deadline=deadline
        if grip_index+1<len(grips):
            suffix_deadline=min(deadline,time.monotonic()+max(0,deadline-time.monotonic())*.85)
        for position,(_,label,prefix,arrival) in enumerate(chosen):
            remaining=suffix_deadline-time.monotonic()
            if remaining<=0 or expanded>=max_expansions:
                break
            # A failed landing must leave time to try a different actual launch
            # state instead of committing the first physical arrival forever.
            # The highest-ranked restored launch gets a substantial contiguous
            # search window. Dividing the same budget among four arrivals and
            # again among search_goal's internal stages starves every branch
            # under parallel CPU contention, despite a short complete solution.
            share=(remaining*.8 if position==0 and len(chosen)>1
                   else remaining/max(1,len(chosen)-position))
            suffix=search_goal(simulator,arrival,goal,deadline=min(suffix_deadline,time.monotonic()+share),
                        max_expansions=max(1,(max_expansions-expanded)//max(1,len(chosen)-position)),
                        max_frames=max_frames-len(prefix),variant=1,allow_unsupported=False)
            expanded+=suffix.expansions;simulated+=suffix.simulated_frames
            stages.append(dict(goal=label,reached=suffix.reached,frames=len(prefix)+len(suffix.actions)))
            if suffix.reached:
                result=SearchResult(prefix+suffix.actions,suffix.state,True,expanded,simulated,"wall_relay_goal")
                result.strategy_details=dict(internal_stages=stages,wall_grips=True)
                return result
            h=heuristic(suffix.state)
            if not suffix.state.dead and h<best_h:
                best_h=h;best=SearchResult(prefix+suffix.actions,suffix.state,False,expanded,simulated,"wall_relay_budget")
    best.expansions=expanded;best.simulated_frames=simulated
    best.strategy_details=dict(internal_stages=stages,wall_grips=True)
    return best


def search_spring_relay(simulator,start,goal,*,deadline,max_expansions,max_frames,variant=1):
    """Engage a real spring launch before searching the original destination.

    A spring can lie below or behind a useful flight. Its approach has a separate
    branch and budget, so temporary geometric regression is not discarded by
    the direct-goal frontier. Launch speed, dash count and timers are preserved.
    """
    from .strategies import search_goal
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    heuristic=Heuristic(data,goal)
    initial_h=heuristic(start)
    candidates=[]
    for entity in data.rooms[start.room].entities:
        spring=spring_goal(start.room,entity)
        if spring is None:
            continue
        distance=math.hypot(start.x-entity.x,start.y-entity.y)
        if distance>420:
            continue
        # Geometry probes rank goals only; they are never search states.
        improvement=initial_h-heuristic(replace(start,x=entity.x,y=entity.y))
        useful_height=max(0,entity.y-goal.y)
        benefit=min(80,useful_height*.35)
        if isinstance(spring,WallSpringGoal) and (goal.x-entity.x)*spring.direction>0:
            benefit+=40
        candidates.append((distance-improvement*.2-benefit,spring.key,spring))
    candidates.sort(key=lambda row:row[:2])
    expanded=simulated=0;stages=[]
    best=SearchResult([],start,False,0,0,"spring_relay_budget")
    best_h=initial_h
    for index,(_,_,spring) in enumerate(candidates[:3]):
        remaining=deadline-time.monotonic()
        if remaining<=0 or expanded>=max_expansions:
            break
        approach=search_leg(simulator,start,spring,
                    deadline=min(deadline,time.monotonic()+min(4.,remaining*.35)),
                    max_expansions=min(4000,max_expansions-expanded),max_frames=max_frames,
                    variant=variant,allow_unsupported=False)
        expanded+=approach.expansions;simulated+=approach.simulated_frames
        stages.append(dict(goal=spring.key,reached=approach.reached,frames=len(approach.actions),
                           actual_launch=bool(approach.reached and spring.achieved(approach.state))))
        if not approach.reached or not spring.achieved(approach.state):
            continue
        if goal.achieved(approach.state):
            result=SearchResult(approach.actions,approach.state,True,expanded,simulated,"spring_relay_goal")
            result.strategy_details=dict(internal_stages=stages,spring_launch=True)
            return result
        remaining=deadline-time.monotonic()
        suffix_deadline=deadline if index+1>=min(3,len(candidates)) else time.monotonic()+remaining*.85
        if remaining<=0 or len(approach.actions)>=max_frames:
            break
        # First preserve the spring's flight instead of immediately asking the
        # landing heuristic to return to a nearer floor and discard its launch.
        flight_deadline=time.monotonic()+max(0,suffix_deadline-time.monotonic())*.8
        suffix=search_leg(simulator,approach.state,goal,deadline=flight_deadline,
                    max_expansions=max(1,max_expansions-expanded),
                    max_frames=max_frames-len(approach.actions),variant=0,
                    allow_unsupported=False)
        expanded+=suffix.expansions;simulated+=suffix.simulated_frames
        if not suffix.reached and time.monotonic()<suffix_deadline and expanded<max_expansions:
            alternative=search_goal(simulator,approach.state,goal,deadline=suffix_deadline,
                    max_expansions=max(1,max_expansions-expanded),
                    max_frames=max_frames-len(approach.actions),variant=variant,
                    allow_unsupported=False)
            expanded+=alternative.expansions;simulated+=alternative.simulated_frames
            if alternative.reached or heuristic(alternative.state)<heuristic(suffix.state):
                suffix=alternative
        actions=approach.actions+suffix.actions
        if suffix.reached:
            result=SearchResult(actions,suffix.state,True,expanded,simulated,"spring_relay_goal")
            result.strategy_details=dict(internal_stages=stages,spring_launch=True)
            return result
        h=heuristic(suffix.state)
        if not suffix.state.dead and h<best_h:
            best_h=h;best=SearchResult(actions,suffix.state,False,expanded,simulated,"spring_relay_budget")
    best.expansions=expanded;best.simulated_frames=simulated
    best.strategy_details=dict(internal_stages=stages,spring_launch=any(stage["actual_launch"] for stage in stages))
    return best


def _coast_support_arrivals(simulator,start,goal,*,deadline,max_frames,stats):
    """Keep physical landings reached while coasting, including temporary solids.

    No geometric progress test removes a higher/backward landing. A spring
    encountered during the coast retains its actual launch and the first later
    landing retains the crumble/falling mechanism's live phase and timer.
    """
    toward=RIGHT if goal.x>start.x else LEFT
    preview=search_tracker(start)
    initial_expanded=stats["expansions"];initial_simulated=stats["simulated_frames"]
    seen=set()
    for mask in (toward,3-toward,0):
        if (time.monotonic()>=deadline
                or stats["expansions"]>=stats.get("max_expansions",float("inf"))):
            break
        stats["expansions"]+=1
        state=start;actions=[];states=[start]
        airborne=not (start.on_ground and start.vy>=0)
        for _ in range(min(120,max_frames)):
            if time.monotonic()>=deadline:
                break
            state=simulator.step(state,mask)
            stats["simulated_frames"]+=1
            actions.append(mask);states.append(state)
            if preview:
                preview.report(lambda:[("alternative",state,states)],
                               stats["expansions"]-initial_expanded,
                               stats["simulated_frames"]-initial_simulated)
            if state.dead or state.room!=start.room:
                break
            grounded=state.on_ground and state.vy>=0
            if goal.achieved(state) or (airborne and grounded):
                key=state_key(state)
                if key not in seen:
                    seen.add(key)
                    remember_segment(start,state,states)
                    yield actions,state,f"coast_landing:{mask}"
                break
            airborne=airborne or not grounded


def search_underhang_relay(simulator,start,goal,*,deadline,max_expansions,max_frames,variant=2):
    """Reach a selected tile under a one-way roof through real edge reversals.

    The independent branch can use a spring and an upper temporary support even
    when those moves increase distance from a low destination. All milestones
    are private search choices; only the original selected tile completes it.
    """
    from .underhang_moves import underhang_edges,underhang_roofs
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    stats=dict(expansions=0,simulated_frames=0,max_expansions=max_expansions)
    stages=[]
    best=SearchResult([],start,False,0,0,"underhang_relay_budget")
    heuristic=Heuristic(data,goal);best_h=heuristic(start)

    def attempt(prefix,arrival,label):
        nonlocal best,best_h
        if len(prefix)>=max_frames or stats["expansions"]>=max_expansions:
            return None
        if goal.achieved(arrival):
            return SearchResult(prefix,arrival,True,stats["expansions"],
                                stats["simulated_frames"],"underhang_relay_goal")
        for actions,end,edge_label in underhang_edges(simulator,arrival,goal,
                    deadline=deadline,max_frames=min(160,max_frames-len(prefix)),stats=stats):
            stages.append(dict(goal=label,edge=edge_label,reached=goal.achieved(end),
                               frames=len(prefix)+len(actions),arrival=dict(x=arrival.x,y=arrival.y)))
            if goal.achieved(end):
                return SearchResult(prefix+actions,end,True,stats["expansions"],
                                    stats["simulated_frames"],"underhang_relay_goal")
            h=heuristic(end)
            if h<best_h:
                best_h=h;best=SearchResult(prefix+actions,end,False,0,0,"underhang_relay_budget")
        return None

    # Cheap coast branches run before a best-first approach can consume the
    # budget on the wrong side of the roof. They also expose real crumble
    # landings absent from static coverage tiles.
    answer=None
    if underhang_roofs(simulator,start,goal):
        answer=attempt([],start,"current_state")
        if answer is None:
            for prefix,arrival,label in _coast_support_arrivals(simulator,start,goal,
                       deadline=deadline,max_frames=max_frames,stats=stats):
                answer=attempt(prefix,arrival,label)
                if answer is not None or stats["expansions"]>=max_expansions:
                    break
        if answer is None:
            room=data.rooms[start.room]
            supports=list(_transient_supports(room,start))+list(coverage_goals(room))
            supports=[support for support in supports if support.y<goal.y-12
                      and 12<math.hypot(start.x-support.x,start.y-support.y)<320]
            supports.sort(key=lambda support:(0 if isinstance(support,TransientSupportGoal) else 1,
                           math.hypot(start.x-support.x,start.y-support.y),support.key))
            for support in supports[:4]:
                remaining=deadline-time.monotonic()
                if remaining<=0 or stats["expansions"]>=max_expansions:
                    break
                first=search_leg(simulator,start,support,
                       deadline=min(deadline,time.monotonic()+min(1.5,remaining*.35)),
                       max_expansions=min(1800,max_expansions-stats["expansions"]),
                       max_frames=max_frames,variant=variant,allow_unsupported=False)
                stats["expansions"]+=first.expansions
                stats["simulated_frames"]+=first.simulated_frames
                stages.append(dict(goal=support.key,reached=first.reached,frames=len(first.actions)))
                if first.reached:
                    answer=attempt(first.actions,first.state,support.key)
                    if answer is not None:
                        break
    if answer is None:
        answer=best
        answer.expansions=stats["expansions"];answer.simulated_frames=stats["simulated_frames"]
    answer.strategy_details=dict(internal_stages=stages,underhang=True)
    return answer
