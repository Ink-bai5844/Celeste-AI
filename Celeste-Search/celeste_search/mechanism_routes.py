"""Bounded capability guidance for continuous mechanism relays.

Only the returned *contact goals* are physical predicates. The small optimistic
graph estimates what a spring, climb or mover ride might enable after contact;
its launch envelopes are never SimStates, executed edges or reachability proof.
Every caller must approach, ride and launch through the actual simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .planner import coverage_goals
from .physics import _truth
from .strategies import spring_goal, usable_spring
from .wall_moves import WallContactGoal, WallGripGoal, _static_faces
from .zip_grips import ZipContactGoal, zip_top_blocked, _contact


@dataclass(frozen=True)
class _Launch:
    x: float
    y: float
    reach: float
    rise: float
    cost: float = 0.


@dataclass(frozen=True)
class _Node:
    goal: object
    family: str
    identity: str
    point: tuple
    launches: tuple
    wait: float = 0.


def _contact_distance(state, goal):
    """Measure an event's real contact interval, not its guide height."""
    x, y = goal.x, goal.y
    if isinstance(goal, (WallContactGoal, ZipContactGoal)):
        bounds = goal.bounds(state)
        if bounds is not None:
            left, top, right, bottom = bounds
            x = left-4 if goal.facing > 0 else right+4
            y = max(top+1, min(state.y, bottom+10))
    return math.dist((state.x,state.y),(x,y))


def held_zip_contacts(simulator, state):
    """Actual held faces, independent of sampled approach-guide geometry.

    These are already achieved physical events, so normal relay ranking omits
    them. A consumer may sample one bounded ride for each unsampled contact;
    it must not reinsert every achieved support or ordinary wall target.
    """
    data = simulator.map_data if hasattr(simulator, "map_data") else getattr(simulator, "map", None)
    room = data.rooms.get(state.room) if data is not None else None
    if room is None or state.dead or state.state != 1 or state.stamina <= 0:
        return []
    records = {row[0]: row for row in state.mechanisms}
    contacts = []
    for entity in room.entities:
        identity = f"{room.name}:{entity.id}"
        if (entity.kind != "zipMover" or not entity.nodes or entity.height < 12
                or identity in state.broken):
            continue
        record = records.get(identity)
        left, top = (round(record[3]), round(record[4])) if record else (entity.x, entity.y)
        x = left-4 if state.facing > 0 else left+entity.width+4
        goal = ZipContactGoal("wall_grip", room.name, x, state.y,
            id=f"relay_zip:{identity}:{state.facing}", identity=identity,
            mechanism_kind="zipMover", facing=state.facing, grip_offset=state.y-top,
            rect=(entity.x,entity.y,entity.x+entity.width,entity.y+entity.height))
        if goal.achieved(state):
            contacts.append(goal)
    return contacts


def _travel(launch, point):
    """Soft capability penalties retain far/high alternatives for backtracking."""
    dx, dy = abs(point[0]-launch.x), point[1]-launch.y
    excess_x = max(0., dx-launch.reach)
    excess_y = max(0., -dy-launch.rise)
    return (math.hypot(dx, dy)*.15 + excess_x*.8 + excess_y*2.
            + (excess_x*excess_x+excess_y*excess_y)*.025 + launch.cost)


def _source(state):
    dash = 1 if state.dashes > 0 else 0
    grounded = state.on_ground or state.state == 1
    rise = (48. if grounded else 0.) + dash*56
    rise += min(100., max(0., -state.vy)*.45)
    rise += min(58., max(0., -state.lift_vy)*.45)
    reach = (88. if grounded else 56.) + dash*72
    reach += min(80., abs(state.lift_vx)*.45)
    return _Launch(state.x, state.y, reach, rise)


def _clear(simulator, state, x, y):
    room = simulator.map.rooms[state.room]
    return (room.x+4 <= x <= room.x+room.width-4
            and room.y-16 <= y <= room.y+room.height
            and not simulator.solid_at(state.room, x, y, state=state))


def _dash_face_samples(state, rect, dx, dy, geometry):
    """Nearest legal contact coordinates, not one point for a whole face.

    Short faces are checked at every integer player coordinate. Long faces
    use a bounded uniform sample plus all relevant obstacle boundaries, so
    a thin opening beside a large block is not lost between uniform samples.
    Geometry supplies query points only; _clear and _hazard_hit remain the
    final admission checks. At most one goal is emitted for each block/face.
    """
    left,top,right,bottom=rect
    if dx:
        low,high=math.floor(top)+1,math.ceil(bottom+11)-1
        preferred=max(low,min(state.y,high))
        fixed=left-8 if dx>0 else right+8
    else:
        low,high=math.floor(left-4)+1,math.ceil(right+4)-1
        preferred=max(low,min(state.x,high))
        fixed=top-4 if dy>0 else bottom+15
    if low>high:
        return ()
    if high-low<=96:
        values=set(range(low,high+1))
    else:
        values={round(low+(high-low)*index/32) for index in range(33)}
        values.update(preferred+offset for offset in (-16,-8,-4,-2,-1,0,1,2,4,8,16))
        for a,b,c,d in geometry:
            if dx and fixed-4<c and fixed+4>a:
                # Standing body: [y-11,y]. Hazard hurtbox also has y-2.
                values.update((b-1,b,b+1,b+2,b+3,d+10,d+11,d+12))
            elif dy and fixed-11<d and fixed>b:
                values.update((a-5,a-4,a-3,c+3,c+4,c+5))
    values.add(preferred)
    return tuple((fixed,value) if dx else (value,fixed)
                 for value in sorted((v for v in values if low<=v<=high),
                                     key=lambda value:(abs(value-preferred),value)))


def dash_break_goals(simulator, state):
    """Propose real block identities even without a preselected corridor.

    The sampled poses below are geometry queries only, never search states.
    Reaching an approach, spending a dash and actually breaking its block are
    separate physical obligations of the action generator.
    """
    from .corridor_routes import DashBlockBreakGoal, _hazard_rect
    room=simulator.map.rooms.get(state.room)
    if room is None or state.dead:
        return []
    blocks=[entity for entity in room.entities if entity.kind=="dashBlock"
            and f"{state.room}:{entity.id}" not in state.broken
            and _truth(entity.attrs.get("canDash",True))]
    if not blocks:
        return []
    actual={identity:rect for identity,_,rect in simulator._mechanism_rects(state.room,state)}
    geometry=list(simulator._static_rects[state.room])+list(actual.values())
    for hazard in simulator._hazards[state.room]:
        bounds=_hazard_rect(hazard)
        owner=simulator._spike_attachments[state.room].get((hazard.kind,hazard.id))
        if owner is not None:
            owner_rect=actual.get(f"{state.room}:{owner.id}")
            if owner_rect is None:
                continue
            ox,oy=owner_rect[0]-owner.x,owner_rect[1]-owner.y
            bounds=tuple(value+(ox if index%2==0 else oy) for index,value in enumerate(bounds))
        geometry.append(bounds)
    goals=[]
    for entity in blocks:
        identity=f"{state.room}:{entity.id}"
        left,top,right,bottom=actual.get(identity,
            (entity.x,entity.y,entity.x+entity.width,entity.y+entity.height))
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            for px,py in _dash_face_samples(state,(left,top,right,bottom),dx,dy,geometry):
                if not _clear(simulator,state,px,py):
                    continue
                pose=replace(state,x=px,y=py,vx=dx,vy=dy,ducking=False)
                if simulator._hazard_hit(pose):
                    continue
                goals.append(DashBlockBreakGoal("dash_block_break",state.room,px,py,
                    id=f"relay_dash_break:{identity}:{dx}:{dy}",identity=identity,
                    rect=(left,top,right,bottom),dash_dx=dx,dash_dy=dy))
                break
    return goals


def _inventory(simulator, state, public_goal):
    # Local import avoids assist_strategies -> relay -> assist_strategies cycles.
    from .assist_strategies import ZipBoardGoal
    from .corridor_routes import RefillPortalGoal
    room = simulator.map.rooms[state.room]
    records = {row[0]:row for row in state.mechanisms}
    cooldowns = dict(state.refill_timers)
    groups = {"spring":[], "zip":[], "refill":[], "support":[], "wall":[], "dash_break":[]}
    for entity in room.entities:
        identity = f"{room.name}:{entity.id}"
        if usable_spring(entity):
            goal = spring_goal(room.name, entity)
            groups["spring"].append(_Node(goal,"spring",identity,(goal.x,goal.y),
                (_Launch(goal.x,goal.y-6,220.,160.,8.),)))
        elif entity.kind == "refill":
            goal = RefillPortalGoal("refill",room.name,entity.x,entity.y+5.5,
                id="relay_refill:"+identity,identity=identity,role="refill",
                pickup_rect=(entity.x-8,entity.y-8,entity.x+8,entity.y+8))
            if _clear(simulator,state,goal.x,goal.y):
                groups["refill"].append(_Node(goal,"refill",identity,(goal.x,goal.y),
                    (_Launch(goal.x,goal.y,150.,64.,5.),), cooldowns.get(identity,0)*.3))
        elif entity.kind == "zipMover" and entity.nodes and identity not in state.broken:
            row = records.get(identity)
            x,y = (round(row[3]),round(row[4])) if row else (entity.x,entity.y)
            endpoint = entity.nodes[0]
            # These points estimate possible launch opportunities, not a mover
            # teleport. Actual phase/timers still belong exclusively to state.
            distance = math.dist((x,y),endpoint)
            phase = row[1] if row else 0
            wait = (row[2]*.2 if row and phase in (1,3) else 0.)
            for facing in (1,-1):
                gx = x-4 if facing>0 else x+entity.width+4
                offset = min(12.,entity.height/2)
                gy = y+offset
                if entity.height<12 or not _clear(simulator,state,gx,gy):
                    continue
                goal = ZipContactGoal("wall_grip",room.name,gx,gy,
                    id=f"relay_zip:{identity}:{facing}",identity=identity,
                    mechanism_kind="zipMover",facing=facing,grip_offset=offset,
                    rect=(entity.x,entity.y,entity.x+entity.width,entity.y+entity.height))
                launches=[]
                for fraction in (0.,.5,1.):
                    px=x+(endpoint[0]-x)*fraction
                    py=y+(endpoint[1]-y)*fraction
                    # Directional horizontal lift extends an outward launch;
                    # upward lift adds rise. No boost is assumed at idle point.
                    moving = fraction>0 and distance>0
                    dx,dy=endpoint[0]-x,endpoint[1]-y
                    outward = (dx>0 and facing<0) or (dx<0 and facing>0)
                    launches.append(_Launch(px-4 if facing>0 else px+entity.width+4,
                        py+offset,160.+(72. if moving and outward else 0.),
                        108.+(58. if moving and dy<0 else 0.),
                        6.+fraction*distance*.1+wait))
                groups["zip"].append(_Node(goal,"zip",identity,(gx,gy),tuple(launches)))
            if not zip_top_blocked(entity,room):
                gx=x+entity.width/2
                if _clear(simulator,state,gx,y):
                    goal=ZipBoardGoal("surface",room.name,gx,y,
                        id="relay_board:"+identity,identity=identity,width=entity.width)
                    launches=tuple(_Launch(gx+(endpoint[0]-x)*f,y+(endpoint[1]-y)*f,
                        220.,160. if endpoint[1]<y else 108.,6.+distance*f*.1+wait)
                        for f in (0.,.5,1.))
                    groups["zip"].append(_Node(goal,"zip",identity,(gx,y),launches))
    for goal in coverage_goals(room):
        if _clear(simulator,state,goal.x,goal.y):
            groups["support"].append(_Node(goal,"support","",(goal.x,goal.y),
                (_Launch(goal.x,goal.y,160.,104.,4.),)))
    for rect,facing in _static_faces(simulator,room):
        left,top,right,bottom=rect
        if bottom-top<16:
            continue
        x=left-4 if facing>0 else right+4
        y=max(top+12,min(state.y-32,bottom-8))
        if not _clear(simulator,state,x,y):
            continue
        goal=WallContactGoal("wall_grip",room.name,x,y,
            id=f"relay_wall:{room.name}:{left:g}:{top:g}:{right:g}:{bottom:g}:{facing}",
            rect=rect,facing=facing,grip_offset=y-top)
        if goal.achieved(state):
            # Once caught, a higher grip is a separate physical climb event.
            # Approaching a different wall must stop at the first useful
            # contact instead of spending stamina to reach this guide height.
            goal=WallGripGoal("wall_grip",room.name,x,y,id=goal.id,
                rect=rect,facing=facing,grip_offset=y-top)
        climb=min(96.,max(0.,y-(top+12)))
        groups["wall"].append(_Node(goal,"wall","",(x,y),
            (_Launch(x,y,160.,104.,5.),_Launch(x,y-climb,160.,104.,5.+climb*.3))))
    for goal in dash_break_goals(simulator,state):
        groups["dash_break"].append(_Node(goal,"dash_break",goal.identity,(goal.x,goal.y),
            (_Launch(goal.x,goal.y,128.,80.,12.),)))
    def order(node):
        return (math.dist((state.x,state.y),node.point)
                + min(math.dist((p.x,p.y),(public_goal.x,public_goal.y)) for p in node.launches)*.35,
                node.goal.key)
    # Family quotas bound graph work without letting a tiled wall erase every
    # spring, mover or recovery opportunity. No physical edge is cached.
    nodes=[]
    for family,cap in (("spring",4),("zip",10),("refill",4),("support",10),("wall",4)):
        nodes.extend(sorted(groups[family],key=order)[:cap])
    # Reserve different destructible identities before adding their alternate
    # faces. One nearby wide block must not erase the next block in a relay.
    blocks={}
    for node in sorted(groups["dash_break"],key=order):
        blocks.setdefault(node.identity,[]).append(node)
    nodes.extend([rows[index] for index in range(4) for rows in blocks.values()
                  if index<len(rows)][:4])
    return nodes


def _rank(simulator,state,public_goal,visited=()):
    if state.dead or state.room not in simulator.map.rooms:
        return []
    nodes=_inventory(simulator,state,public_goal)
    target=(public_goal.x,public_goal.y)
    # Reverse shortest paths over <=36 optimistic capability nodes. Contact
    # with the next node is still mandatory in the physical solver.
    costs=[min(_travel(launch,target) for launch in node.launches) for node in nodes]
    edges=[[math.inf if i==j or (node.identity and node.identity==other.identity)
            else min(_travel(p,other.point) for p in node.launches)+other.wait
            for j,other in enumerate(nodes)] for i,node in enumerate(nodes)]
    settled=set()
    for _ in nodes:
        j=min((i for i in range(len(nodes)) if i not in settled),key=lambda i:costs[i])
        settled.add(j)
        for i in range(len(nodes)):
            if i not in settled:
                costs[i]=min(costs[i],edges[i][j]+costs[j])
    used={getattr(item,"key",item) for item in visited}
    source=_source(state)
    result=[]
    for node,cost in zip(nodes,costs):
        if node.goal.key in used or node.goal.achieved(state):
            continue
        score=_travel(source,node.point)+node.wait+cost
        result.append(dict(goal=node.goal,score=score,family=node.family,
                           identity=node.identity,remaining_cost=cost))
    return sorted(result,key=lambda row:(row["score"],row["goal"].key))


def relay_candidate_details(simulator,state,public_goal,*,visited=(),limit=8):
    """Rank contact goals; preserve a bounded mechanism and recovery alternative."""
    if limit<=0:
        return []
    rows=_rank(simulator,state,public_goal,visited)
    # A current ride is an opportunity to transfer, not a reason to reacquire
    # the opposite face of the same mover. Reserve a continuation in its real
    # lift/path direction before nearby return walls consume the small quota.
    contacts=held_zip_contacts(simulator,state) if rows else []
    if rows:
        contacts.extend(node.goal for node in _inventory(simulator,state,public_goal)
            if node.family=="zip" and not isinstance(node.goal,WallGripGoal)
            and node.goal.achieved(state))
    promoted=[]
    if not contacts and (state.on_ground and state.vy >= 0 or state.state == 1):
        # Optimistic suffix costs can favor a distant mover behind several
        # obstacles. Give the nearest local mechanism a first physical probe;
        # retain the optimistic alternatives for backtracking if it fails.
        # After a temporary wall catch, a nearby already-triggered mover is
        # also a valid next event, including a return to its opposite side.
        active={record[0] for record in state.mechanisms if record[1] != 0}
        onward=1 if public_goal.x >= state.x else -1
        local=[row for row in rows if row["family"] in ("spring","zip")
               and (state.on_ground or row["identity"] in active or
                    abs(public_goal.x-state.x)>=48 and (row["goal"].x-state.x)*onward>0)
               and _contact_distance(state,row["goal"])
                   <= (112 if state.on_ground else 64 if row["identity"] in active else 96)]
        if local:
            promoted.append(min(local,key=lambda row:(
                _contact_distance(state,row["goal"]),row["score"])))
            # A returning mover and an unused onward mover are different
            # capabilities. Reserving only the closest one can remove the
            # descent/transfer needed to get around a thick wall entirely.
            # Keep each class bounded; the physical search still has to
            # reach its actual contact and can backtrack to either choice.
            if state.state == 1:
                for candidates in (
                    [row for row in local if row["identity"] in active],
                    [row for row in local if row["identity"] not in active
                     and (row["goal"].x-state.x)*onward>0],
                ):
                    if candidates:
                        candidate=min(candidates,key=lambda row:(
                            _contact_distance(state,row["goal"]),row["score"]))
                        if candidate not in promoted:
                            promoted.append(candidate)
        # A nearby forward refill can be the next launch in a continuous
        # relay even while the current contact still has a full dash: flying
        # there may spend it before the pickup. Reserve one physical probe,
        # not a virtual resource gain, ahead of speculative return walls.
        cooldowns=dict(state.refill_timers)
        toward=(public_goal.x-state.x,public_goal.y-state.y)
        refills=[row for row in rows if row["family"]=="refill"
                 and not getattr(simulator,"no_dash",False)
                 and cooldowns.get(row["identity"],0)<=0
                 and _contact_distance(state,row["goal"])<=(112 if state.on_ground else 96)
                 and (row["goal"].x-state.x)*toward[0]
                     +(row["goal"].y-state.y)*toward[1]>0]
        if refills:
            promoted.append(min(refills,key=lambda row:(
                _contact_distance(state,row["goal"]),row["score"])))
        if state.state == 1:
            # The nearest facing wall is a useful retreat/relay even when
            # the public goal lies the other way. Preserve its first contact
            # before speculative high grips on the same blocked wall.
            catches=[row for row in rows if row["family"]=="wall"
                     and (row["goal"].x-state.x)*row["goal"].facing>4
                     and _contact_distance(state,row["goal"])<=96]
            if catches:
                promoted.append(min(catches,key=lambda row:(
                    _contact_distance(state,row["goal"]),row["score"])))
    if contacts:
        identities={contact.identity for contact in contacts}
        entities={f"{state.room}:{e.id}":e for e in simulator.map.rooms[state.room].entities}
        directions=[]
        for identity in identities:
            e=entities[identity]
            dx,dy=e.nodes[0][0]-e.x,e.nodes[0][1]-e.y
            directions.append((state.lift_vx if abs(state.lift_vx)>1 else dx,
                               state.lift_vy if abs(state.lift_vy)>1 else dy))
        # A nearby exposed wall can be a temporary hold before returning to
        # the mover on its useful side. Reserve an approach-facing wall above
        # the current feet, before distant optimistic mechanism transfers.
        onward=1 if public_goal.x >= state.x else -1
        # A short retreat to a facing static wall can hold position while the
        # mover clears an opening. Its usefulness does not depend on pointing
        # toward the eventual public destination. Keep the existing onward
        # and other-mechanism choices alongside this one bounded alternative.
        catches=[row for row in rows if row["family"]=="wall"
            and 4 < (row["goal"].x-state.x)*row["goal"].facing
            and _contact_distance(state,row["goal"]) <= 80
            and row["goal"].rect[1] < state.y]
        if catches:
            promoted.append(min(catches,key=lambda row:(
                _contact_distance(state,row["goal"]),row["score"])))
        nearby_walls=[row for row in rows if row["family"]=="wall"
            and 4 < (row["goal"].x-state.x)*onward <= 80
            and (row["goal"].x-state.x)*row["goal"].facing > 0
            and row["goal"].rect[1] < state.y
            and abs(row["goal"].y-state.y) <= 80]
        if nearby_walls:
            nearest=min(nearby_walls,key=lambda row:
                math.dist((state.x,state.y),(row["goal"].x,row["goal"].y)))
            if nearest not in promoted:
                promoted.append(nearest)
        transfers=[row for row in rows if row["family"]=="zip" and row["identity"] not in identities
                   and (any((row["goal"].x-state.x)*dx+(row["goal"].y-state.y)*dy>0
                            for dx,dy in directions)
                        or abs(public_goal.x-state.x)>=48
                        and (row["goal"].x-state.x)*onward>0
                        and _contact_distance(state,row["goal"])<=128)]
        if transfers:
            # The optimistic launch envelope can make a mover's far side
            # look cheaper, although reaching it costs another climb jump.
            # Probe the actual approach-facing contact first; keep far-side
            # choices in the ranked remainder for backtracking.
            promoted.append(min(transfers,key=lambda row:(
                (row["goal"].x-state.x)*getattr(row["goal"],"facing",0)<=0,
                _contact_distance(state,row["goal"]),row["score"])))
        if any(dy<0 for _,dy in directions):
            supports=[row for row in rows if row["family"]=="support" and row["goal"].y<state.y]
            if supports:
                # A nearby landing can restore resources before a further
                # ascent. This is ordering only, never a reach-height cutoff.
                promoted.append(min(supports,key=lambda row:
                    (state.y-row["goal"].y)+abs(state.x-row["goal"].x)*.5))
    # Opening a nearby obstacle is itself a useful event. A purely optimistic
    # suffix score can otherwise retain only a farther block behind it. Keep
    # different block identities in this small active-action allowance.
    block_ids=set()
    for row in sorted((row for row in rows if row["family"]=="dash_break"),
                      key=lambda row:(_contact_distance(state,row["goal"]),row["score"])):
        if row["identity"] in block_ids:
            continue
        promoted.append(row)
        block_ids.add(row["identity"])
        if len(block_ids)>=min(2,max(1,limit//3)):
            break
    rows=promoted+[row for row in rows if row not in promoted]
    selected=rows[:limit]
    # Reserve useful families when many nearby wall contacts share one score.
    reserved={row["goal"].key for row in promoted[:limit]}
    for families in (("spring","zip"),("support","refill"),("dash_break",)):
        present=next((row for row in selected if row["family"] in families),None)
        if present is not None:
            reserved.add(present["goal"].key)
        elif selected:
            alternative=next((row for row in rows if row["family"] in families),None)
            if alternative is not None and limit>=3:
                replace_at=next((i for i in reversed(range(len(selected)))
                                 if selected[i]["goal"].key not in reserved),None)
                if replace_at is not None:
                    selected[replace_at]=alternative
                    reserved.add(alternative["goal"].key)
    return selected


def relay_candidates(simulator,state,public_goal,*,visited=(),limit=8):
    return [row["goal"] for row in relay_candidate_details(
        simulator,state,public_goal,visited=visited,limit=limit)]


def relay_priority(simulator,state,public_goal):
    """Optimistic continuation score for an actual arrival, never an acceptance."""
    if state.dead or state.room not in simulator.map.rooms:
        return math.inf
    direct=_travel(_source(state),(public_goal.x,public_goal.y))
    rows=_rank(simulator,state,public_goal)
    return min([direct]+[row["score"] for row in rows])
