"""Body-clearance topology proposals, deliberately separate from physics proof.

The grid describes where the standing player's feet can geometrically be. It
does not assume that gravity, stamina or a jump can carry the player between
those cells. A caller must solve the portals from real states and replay the
entire original user route before any candidate becomes executable.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import OrderedDict, deque
import hashlib
import heapq
import math
import time

from .planner import Goal, coverage_goals
from .physics import _truth
from .wall_moves import WallGripGoal, _static_faces
from .zip_grips import ZipContactGoal

_GRID_CACHE = OrderedDict()
_GRID_CACHE_LIMIT = 6
_GRID_CACHE_HITS = 0
_GRID_CACHE_MISSES = 0
_CANDIDATE_CACHE = OrderedDict()
_CANDIDATE_POOL_LIMIT = 24
_CONDITIONAL_RESERVE = 6


class CorridorCandidates(list):
    """A page plus bounded-discovery status; an empty timeout is not exhaustion."""
    def __init__(self, values=(), *, complete=False, reason="generation_time_budget",
                 next_offset=0, pool_size=0, generation_seconds=0., discovery_cursor=0,
                 continuation=None):
        super().__init__(values)
        self.complete = complete
        self.reason = reason
        self.next_offset = next_offset
        self.pool_size = pool_size
        self.generation_seconds = generation_seconds
        self.discovery_cursor = discovery_cursor
        self.continuation = continuation


def corridor_cache_info():
    return dict(entries=len(_GRID_CACHE),limit=_GRID_CACHE_LIMIT,
                hits=_GRID_CACHE_HITS,misses=_GRID_CACHE_MISSES,
                cells=sum(len(grid.free) for grid in _GRID_CACHE.values()))


def clear_corridor_cache():
    global _GRID_CACHE_HITS, _GRID_CACHE_MISSES
    _GRID_CACHE.clear();_GRID_CACHE_HITS=_GRID_CACHE_MISSES=0
    _CANDIDATE_CACHE.clear()


@dataclass(frozen=True)
class CorridorPortalGoal(Goal):
    bounds: tuple[float, float, float, float] = (0., 0., 0., 0.)
    role: str = "passage"

    def achieved(self, state):
        left, top, right, bottom = self.bounds
        return (not state.dead and state.room == self.room
                and left <= state.x <= right and top <= state.y <= bottom
                and (self.role != "support" or state.on_ground and state.vy >= 0)
                and (self.role != "refill" or state.dashes > 0))


@dataclass(frozen=True)
class RefillPortalGoal(CorridorPortalGoal):
    """Accept the actual pickup boundary and this entity's restored resource."""
    identity: str = ""
    pickup_rect: tuple[float,float,float,float] = (0.,0.,0.,0.)

    def achieved(self,state):
        if state.dead or state.room!=self.room or state.dashes<=0:
            return False
        if not any(key==self.identity and timer>0 for key,timer in state.refill_timers):
            return False
        hurtbox=(state.x-4,state.y-(6 if state.ducking else 11),state.x+4,state.y-2)
        return _overlap(hurtbox,self.pickup_rect)


@dataclass(frozen=True)
class FallingBlockTriggerGoal(WallGripGoal):
    """An actual grip that has started this particular block's lifecycle."""
    def achieved(self, state):
        record = next((row for row in state.mechanisms if row[0] == self.identity), None)
        return (record is not None and record[1] not in (0, 4)
                and super().achieved(state))


@dataclass(frozen=True)
class ZipMoverTriggerGoal(ZipContactGoal):
    """The named mover must be triggered while the player really grips it."""
    def achieved(self, state):
        record=next((row for row in state.mechanisms if row[0]==self.identity),None)
        return record is not None and record[1]!=0 and super().achieved(state)


@dataclass(frozen=True)
class MechanismPassageGoal(CorridorPortalGoal):
    """The player must actually enter the opened passage before continuing."""
    identity: str = ""


@dataclass(frozen=True)
class DashBlockApproachGoal(CorridorPortalGoal):
    """A body-clear approach while the block and its hazards still exist."""
    identity: str = ""


@dataclass(frozen=True)
class DashBlockBreakGoal(Goal):
    """Require the simulator's real collision result, never an omitted solid."""
    identity: str = ""
    rect: tuple = ()
    clearance_box: tuple = ()
    dash_dx: int = 0
    dash_dy: int = 0

    def achieved(self, state):
        return not state.dead and self.identity in state.broken


@dataclass(frozen=True)
class DashBlockPassageGoal(MechanismPassageGoal):
    """Breaking a block alone is not proof of passing through its opening."""
    def achieved(self, state):
        return self.identity in state.broken and super().achieved(state)


@dataclass(frozen=True)
class MechanismClearanceGoal(Goal):
    """A conditional passage, opened only by the observed simulated mechanism.

    ``x,y`` is a reachable approach-side grip position, never the block centre.
    This condition does not move the player or the block. The solver must emit
    and check real release/hold/retreat inputs until the passage is clear.
    """
    identity: str = ""
    rect: tuple = ()
    clearance_box: tuple = ()
    facing: int = 1
    attached_hazards: tuple = ()
    mechanism_kind: str = "fallingBlock"

    def waypoint(self, state):
        return self.x, self.y

    def achieved(self, state):
        if state.dead or state.room != self.room:
            return False
        record = next((row for row in state.mechanisms if row[0] == self.identity), None)
        if record is None or record[1] == 0:
            return False
        if self.mechanism_kind=="fallingBlock" and record[1] == 4:
            return True
        left, top, right, bottom = self.rect
        x, y = round(record[3]), round(record[4])
        actual = (x, y, x+right-left, y+bottom-top)
        if _overlap(actual, self.clearance_box):
            return False
        return not any(_overlap((a+x-left, b+y-top, c+x-left, d+y-top), self.clearance_box)
                       for a,b,c,d in self.attached_hazards)


@dataclass(frozen=True)
class ZipMoverPassageGoal(MechanismPassageGoal):
    """Cross the open former footprint toward the next leg, at a real height.

    The guide point remains useful for drawing and search. A carried player
    can already have crossed this region when the clearance stage completes;
    returning to one arbitrary guide height must not undo that real progress.
    """
    occupied_rect: tuple = ()
    footprints: tuple = ()
    crossing_axis: int = 0
    crossing_sign: int = 1
    clearance: MechanismClearanceGoal | None = None

    def achieved(self,state):
        if (state.dead or state.room!=self.room or self.clearance is None
                or self.clearance.identity!=self.identity
                or not self.clearance.achieved(state)
                or len(self.occupied_rect)!=4 or self.crossing_axis not in (0,1)
                or self.crossing_sign not in (-1,1)):
            return False
        height=6 if state.ducking else 11
        body=(state.x-4,state.y-height,state.x+4,state.y)
        if not any(_overlap(body,rect) for rect in self.footprints):
            return False
        midpoint=(self.occupied_rect[self.crossing_axis]+
                  self.occupied_rect[self.crossing_axis+2])/2
        center=state.x if self.crossing_axis==0 else state.y-height/2
        return (center-midpoint)*self.crossing_sign>=0


@dataclass(frozen=True)
class CorridorRoute:
    id: str
    room: str
    portals: tuple[Goal, ...]
    points: tuple[tuple[float, float], ...]
    signature: tuple[str, ...]
    length: float
    conditional_on: tuple[str, ...] = ()

    def to_dict(self):
        return dict(id=self.id, room=self.room, coordinates="world_feet", status="pending",
                    points=[dict(room=self.room, x=x, y=y) for x, y in self.points],
                    portals=[dict(id=goal.key, kind=goal.kind, room=goal.room, x=goal.x, y=goal.y,
                                  **(dict(identity=goal.identity, rect=goal.rect,
                                          clearance_box=goal.clearance_box,
                                          dash_dx=goal.dash_dx,dash_dy=goal.dash_dy,
                                          condition="actual_dash_break")
                                     if isinstance(goal,DashBlockBreakGoal) else
                                     dict(identity=goal.identity,condition="occupied_block_approach")
                                     if isinstance(goal,DashBlockApproachGoal) else
                                     dict(identity=goal.identity,condition="actual_break_and_player_passage")
                                     if isinstance(goal,DashBlockPassageGoal) else
                                     dict(identity=goal.identity, clearance_box=goal.clearance_box,
                                          rect=goal.rect, facing=goal.facing,
                                          mechanism_kind=goal.mechanism_kind,
                                          condition="actual_mover_and_hazard_clearance")
                                     if isinstance(goal,MechanismClearanceGoal) else
                                     dict(identity=goal.identity, mechanism_kind=goal.mechanism_kind,
                                          condition="actual_grip_and_triggered")
                                     if isinstance(goal,(FallingBlockTriggerGoal,ZipMoverTriggerGoal)) else
                                     dict(identity=goal.identity,condition="actual_open_footprint_crossing",
                                          occupied_rect=goal.occupied_rect,footprints=goal.footprints,
                                          crossing_axis=goal.crossing_axis,crossing_sign=goal.crossing_sign)
                                     if isinstance(goal,ZipMoverPassageGoal) else
                                     dict(identity=goal.identity, condition="actual_player_inside_passage")
                                     if isinstance(goal,MechanismPassageGoal) else
                                     dict(identity=goal.identity, condition="actual_refill_pickup")
                                     if isinstance(goal,RefillPortalGoal) else {}))
                             for goal in self.portals],
                    signature=list(self.signature), length=self.length,
                    conditional_on=list(self.conditional_on),
                    conditions=[dict(kind="dash_break",identity=goal.identity,room=goal.room,
                                     rect=goal.rect,can_dash=True,approach=dict(x=goal.x,y=goal.y),
                                     dash_direction=dict(x=goal.dash_dx,y=goal.dash_dy))
                                for goal in self.portals if isinstance(goal,DashBlockBreakGoal)]+
                               [dict(kind="zip_clearance",identity=goal.identity,room=goal.room,
                                     rect=goal.rect,clearance_box=goal.clearance_box,
                                     approach=dict(x=goal.x,y=goal.y),
                                     condition="actual_trigger_clearance_and_passage",
                                     physically_verified=False)
                                for goal in self.portals if isinstance(goal,MechanismClearanceGoal)
                                and goal.mechanism_kind=="zipMover"],
                    geometry_status="requires_mechanism_change" if self.conditional_on else "current_state_clear",
                    physically_verified=False)


def _overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _hazard_rect(entity):
    x,y=entity.x,entity.y
    if entity.kind == "spikesUp": return x,y-3,x+max(8,entity.width),y
    if entity.kind == "spikesDown": return x,y,x+max(8,entity.width),y+3
    if entity.kind == "spikesLeft": return x-3,y,x,y+max(8,entity.height)
    if entity.kind == "spikesRight": return x,y,x+3,y+max(8,entity.height)
    return x-6,y-6,x+6,y+6


class _ClearanceGrid:
    """Static body/hurtbox clearance and directed one-way-platform crossings."""
    def __init__(self, simulator, room, step, deadline, *, extra_solids=(), hazard_rects=None):
        self.room, self.step, self.deadline = room, step, deadline
        self.solids = simulator._static_rects[room.name] + tuple(extra_solids)
        self.platforms = simulator._platforms[room.name]
        # Conservative hazard geometry; no velocity-dependent exemption.
        self.hazards = (list(hazard_rects) if hazard_rects is not None else
                        [_hazard_rect(e) for e in simulator._hazards[room.name]])
        self.nx = int((room.width-8)//step)+1
        self.ny = int((room.height-12)//step)+1
        self.free = set()
        self.edges = {}
        self._segments = {}
        self._paths = OrderedDict()
        self._next_cell = 0
        self.complete = False
        self.resume_build(deadline)

    def resume_build(self, deadline):
        """Continue the exact unfinished grid instead of scanning it again."""
        self.deadline=deadline
        count=self.nx*self.ny
        while self._next_cell<count:
            if self._next_cell%16==0 and time.monotonic()>=deadline:
                return
            cell=(self._next_cell%self.nx,self._next_cell//self.nx)
            if self.clear(self.point(cell)):
                self.free.add(cell)
            self._next_cell+=1
        self.complete=True

    def point(self, cell):
        return self.room.x+4+cell[0]*self.step, self.room.y+12+cell[1]*self.step

    def clear(self, point):
        x, y = point
        if not (self.room.x+4 <= x <= self.room.x+self.room.width-4
                and self.room.y+11 <= y <= self.room.y+self.room.height):
            return False
        body=(x-4,y-11,x+4,y)
        hurt=(x-4,y-11,x+4,y-2)
        return not (any(_overlap(body,solid) for solid in self.solids)
                    or any(_overlap(hurt,hazard) for hazard in self.hazards))

    def segment_clear(self, before, after):
        key=(tuple(before),tuple(after))
        if key in self._segments:
            return self._segments[key]
        dx,dy=after[0]-before[0],after[1]-before[1]
        if dy > 0:
            for left,top,right,_ in self.platforms:
                if before[1] <= top < after[1]:
                    x=before[0]+dx*(top-before[1])/dy
                    if x+4 > left and x-4 < right:
                        if len(self._segments)<8192:
                            self._segments[key]=False
                        return False
        count=max(1,math.ceil(max(abs(dx),abs(dy))/2))
        result=all(self.clear((before[0]+dx*i/count,before[1]+dy*i/count))
                   for i in range(count+1))
        if len(self._segments)<8192:
            self._segments[key]=result
        return result

    def nearest(self, point, *, radius=12, from_point=True):
        cx=round((point[0]-self.room.x-4)/self.step)
        cy=round((point[1]-self.room.y-12)/self.step)
        reach=math.ceil(radius/self.step)
        choices=sorted((math.dist(point,self.point((x,y))),x,y)
                       for x in range(cx-reach,cx+reach+1)
                       for y in range(cy-reach,cy+reach+1)
                       if (x,y) in self.free and math.dist(point,self.point((x,y))) <= radius)
        for _,x,y in choices:
            segment=(point,self.point((x,y))) if from_point else (self.point((x,y)),point)
            if self.segment_clear(*segment):
                return x,y
        return None

    def neighbors(self, cell):
        if cell not in self.edges:
            x,y=cell;result=[]
            for dx,dy in ((0,-1),(-1,0),(1,0),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)):
                nxt=(x+dx,y+dy)
                if nxt not in self.free:
                    continue
                if dx and dy and ((x+dx,y) not in self.free or (x,y+dy) not in self.free):
                    continue
                if self.segment_clear(self.point(cell),self.point(nxt)):
                    result.append((nxt,self.step*math.hypot(dx,dy)))
            self.edges[cell]=tuple(result)
        return self.edges[cell]

    def path(self, start, end, penalties=None, forbidden=frozenset(), *, cache_tag=None):
        if not self.complete or start is None or end is None:
            return []
        # Only deterministic geometry requests are memoized. Penalized paths
        # supply their recipe index; arbitrary caller penalty dictionaries keep
        # the old uncached behavior. No physics states are stored in this grid.
        key=(start,end,cache_tag,frozenset(forbidden)) if not penalties or cache_tag is not None else None
        record=self._paths.get(key) if key is not None else None
        if record is not None and record["done"]:
            return record["result"]
        if time.monotonic()>=self.deadline:
            return []
        if record is None:
            record=dict(done=False,queue=[(0.,0.,start)],cost={start:0.},parent={})
            if key is not None:
                self._paths[key]=record
                while len(self._paths)>96:
                    self._paths.popitem(last=False)
        queue,cost,parent=record["queue"],record["cost"],record["parent"]
        while queue and time.monotonic() < self.deadline:
            _,spent,cell=heapq.heappop(queue)
            if spent != cost[cell]:
                continue
            if cell == end:
                result=[cell]
                while cell != start:
                    cell=parent[cell];result.append(cell)
                result=result[::-1]
                record.clear();record.update(done=True,result=result)
                return result
            for nxt,distance in self.neighbors(cell):
                if nxt in forbidden:
                    continue
                value=spent+distance*(1+(penalties or {}).get(nxt,0.))
                if value < cost.get(nxt,math.inf):
                    cost[nxt]=value;parent[nxt]=cell
                    heuristic=math.dist(nxt,end)*self.step
                    heapq.heappush(queue,(value+heuristic,value,nxt))
        if not queue:
            record.clear();record.update(done=True,result=[])
        return []

    def obstacle_centers(self):
        """Interior blocked components distinguish opposite obstacle bypasses."""
        if hasattr(self,"_centers"):
            return self._centers
        blocked={(x,y) for x in range(self.nx) for y in range(self.ny)}-self.free
        result=[]
        while blocked:
            seed=min(blocked);blocked.remove(seed);queue=[seed];component=[];border=False
            for x,y in queue:
                component.append((x,y));border |= x in (0,self.nx-1) or y in (0,self.ny-1)
                for cell in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if cell in blocked:
                        blocked.remove(cell);queue.append(cell)
            if not border and len(component)>=2:
                x=sum(cell[0] for cell in component)/len(component)
                y=sum(cell[1] for cell in component)/len(component)
                result.append(self.point((x,y)))
        self._centers=result
        return result


def _clearance_grid(simulator, room, step, deadline):
    """Bounded static geometry cache; no state, route, action or deadline reuse."""
    global _GRID_CACHE_HITS, _GRID_CACHE_MISSES
    key=(room.name,room.x,room.y,room.width,room.height,step,
         simulator._static_rects[room.name],simulator._platforms[room.name],
         tuple((e.kind,e.x,e.y,e.width,e.height) for e in simulator._hazards[room.name]))
    cached=_GRID_CACHE.pop(key,None)
    if cached is None:
        _GRID_CACHE_MISSES+=1
        cached=_ClearanceGrid(simulator,room,step,deadline)
        if cached.complete and len(cached.free)<=25000:
            _GRID_CACHE[key]=cached
            while len(_GRID_CACHE)>_GRID_CACHE_LIMIT:
                _GRID_CACHE.popitem(last=False)
    else:
        _GRID_CACHE_HITS+=1
        _GRID_CACHE[key]=cached
    # Share geometry and deterministic adjacency only. Concurrent callers get
    # independent deadlines; a timeout never makes a partial grid reusable.
    result=object.__new__(_ClearanceGrid)
    result.__dict__.update(cached.__dict__)
    result.deadline=deadline
    result._paths=OrderedDict()
    return result


def _signature(points, centers):
    result=[]
    for label,cx,cy in centers:
        angles=[math.atan2(y-cy,x-cx) for x,y in points]
        winding=sum(math.atan2(math.sin(b-a),math.cos(b-a)) for a,b in zip(angles,angles[1:]))
        turns=round((winding-(angles[-1]-angles[0]))/(2*math.pi))
        result.append(f"{label}:{turns}")
    return tuple(result)


def _approach_point(room, goal):
    if getattr(goal,"transition_to",""):
        x=getattr(goal,"tile_x",goal.x)+getattr(goal,"tile_size",8)/2
        y=getattr(goal,"tile_y",goal.y)+getattr(goal,"tile_size",8)/2+5.5
        edge=goal.edge
        bounds=getattr(goal,"arrival_bounds",())
        if len(bounds)==4:
            if edge in {"top","bottom"}:
                x=(bounds[0]+bounds[2])/2
            else:
                y=(bounds[1]+bounds[3])/2+5.5
        if edge=="top": y=room.y+12
        elif edge=="bottom": y=room.y+room.height-4
        elif edge=="left": x=room.x+4
        elif edge=="right": x=room.x+room.width-4
        return x,y
    return goal.x,goal.y


def _anchors(simulator, room, grid):
    if hasattr(grid,"_anchors"):
        return grid._anchors
    anchors=[]
    for goal in coverage_goals(room):
        if grid.clear((goal.x,goal.y)):
            anchors.append(CorridorPortalGoal("surface",room.name,goal.x,goal.y,
                id=f"corridor_support:{room.name}:{goal.x:g}:{goal.y:g}",
                bounds=(goal.x-4,goal.y-1,goal.x+4,goal.y+1),role="support"))
    for rect,facing in _static_faces(simulator,room):
        left,top,right,bottom=rect
        if bottom-top<16:
            continue
        x=left-4 if facing>0 else right+4
        heights={top+min(12,(bottom-top)/2),bottom+8}
        heights.update(top+12+48*i for i in range(1,math.ceil((bottom-top-12)/48)))
        for y in sorted(heights):
            if grid.clear((x,y)):
                anchors.append(WallGripGoal("wall_grip",room.name,x,y,
                    id=f"corridor_wall:{room.name}:{left:g}:{top:g}:{right:g}:{bottom:g}:{facing}:{y:g}",
                    rect=rect,facing=facing,grip_offset=y-top))
    for entity in room.entities:
        if entity.kind=="refill":
            x,y=entity.x,entity.y+5.5
            if grid.clear((x,y)):
                anchors.append(RefillPortalGoal("refill",room.name,x,y,
                    id=f"corridor_refill:{room.name}:{entity.id}",
                    bounds=(x-12,entity.y-6,x+12,entity.y+19),role="refill",
                    identity=f"{room.name}:{entity.id}",
                    pickup_rect=(entity.x-8,entity.y-8,entity.x+8,entity.y+8)))
    grid._anchors=sorted(anchors,key=lambda goal:goal.key)
    return grid._anchors


def _simplify(grid, points):
    result=[points[0]];index=0
    while index<len(points)-1:
        last=index+1
        for candidate in range(index+2,len(points)):
            if not grid.segment_clear(points[index],points[candidate]):
                break
            last=candidate
        result.append(points[last]);index=last
    return result


def _portals(grid, path, anchors, forced, max_portals):
    points=[grid.point(cell) for cell in path]
    bends=_simplify(grid,points)[1:-1]
    progress=[0.]
    for a,b in zip(points,points[1:]): progress.append(progress[-1]+math.dist(a,b))
    candidates=[]
    for goal in anchors:
        if (goal.kind=="surface" and goal not in forced and abs(goal.y-points[0][1])<4
                and math.dist((goal.x,goal.y),points[0])<48):
            continue  # The current support/run-up is not another corridor stage.
        distance,index=min((math.dist((goal.x,goal.y),point),i) for i,point in enumerate(points))
        if isinstance(goal,WallGripGoal) and goal.rect[3]-goal.rect[1]<32:
            left,top,right,_=goal.rect
            inside=(left+4 if goal.facing>0 else right-4,top)
            local_dy=points[min(len(points)-1,index+4)][1]-points[max(0,index-4)][1]
            if local_dy<0 and not grid.clear(inside):
                continue  # An ascending cul-de-sac below an overhanging ceiling.
            if not bends or min(math.dist((goal.x,goal.y),point) for point in bends)>28:
                continue  # Do not demand contact with every short wall passed in open air.
        if ((distance<=24 or goal in forced) and 12<progress[index]<progress[-1]-12
                and grid.segment_clear(points[index],(goal.x,goal.y))
                and grid.segment_clear((goal.x,goal.y),points[index])):
            priority=0 if goal in forced else 1 if goal.kind=="refill" else 2 if goal.kind=="surface" else 3
            candidates.append((priority,distance,index,goal))
    chosen=[]
    for priority,distance,index,goal in sorted(candidates,key=lambda row:(row[0],row[1],row[2],row[3].key)):
        if goal in forced or all(abs(progress[index]-progress[old])>=28 for old,_ in chosen):
            chosen.append((index,goal))
        if len(chosen)>=max_portals:
            break
    # Preserve unavoidable narrow turns even when no support is nearby. Wide
    # open-space bends do not become arbitrary mandatory airborne waypoints.
    for point in _simplify(grid,points)[1:-1]:
        index=min(range(len(points)),key=lambda i:math.dist(point,points[i]))
        if len(chosen)>=max_portals or any(abs(progress[index]-progress[old])<28 for old,_ in chosen):
            continue
        x,y=point
        if sum(grid.clear((x+dx,y+dy)) for dx,dy in ((-8,0),(8,0),(0,-8),(0,8)))>=4:
            continue
        chosen.append((index,CorridorPortalGoal("corridor",grid.room.name,x,y,
                       id=f"corridor_passage:{grid.room.name}:{x:g}:{y:g}",
                       bounds=(x-4,y-4,x+4,y+4))))
    ordered=[]
    for _,goal in sorted(chosen,key=lambda row:(row[0],row[1].key)):
        if ordered and isinstance(goal,WallGripGoal) and isinstance(ordered[-1],WallGripGoal) \
                and (goal.rect,goal.facing)==(ordered[-1].rect,ordered[-1].facing):
            ordered[-1]=goal  # One continuous face, not a mandatory stop at every height.
        else:
            ordered.append(goal)
    return tuple(ordered)


def _display_points(grid, points, portals):
    """Reconnect mandatory portals with directed clear edges, never shortcut them."""
    result=[points[0]];visited=set();current=grid.nearest(points[0]);segments=[]
    milestones=[]
    for point in _simplify(grid,points)[1:-1]:
        if any(math.dist(point,(portal.x,portal.y))<=12 for portal in portals):
            continue
        index=min(range(len(points)),key=lambda i:math.dist(point,points[i]))
        milestones.append((index,0,point))
    for portal in portals:
        point=(portal.x,portal.y)
        index=min(range(len(points)),key=lambda i:math.dist(point,points[i]))
        milestones.append((index,1,point))
    for _,_,point in sorted(milestones):
        target=grid.nearest(point,from_point=False)
        path=grid.path(current,target,forbidden=visited)
        if not path:
            return ()
        visited.update(path[:-1]);segments.append((path,point));current=target
    path=grid.path(current,grid.nearest(points[-1]),forbidden=visited)
    if not path:
        return ()
    segments.append((path,points[-1]))
    for path,exact in segments:
        piece=_simplify(grid,[grid.point(cell) for cell in path])
        if result[-1]!=piece[0] and len(piece)>1 and grid.segment_clear(result[-1],piece[1]):
            piece=piece[1:]
        if not grid.segment_clear(result[-1],piece[0]):
            return ()
        result.extend(piece if result[-1]!=piece[0] else piece[1:])
        if result[-1]!=exact:
            if not grid.segment_clear(result[-1],exact):
                return ()
            result.append(exact)
    if not all(grid.segment_clear(a,b) for a,b in zip(result,result[1:])):
        return ()
    return tuple(point for index,point in enumerate(result) if not index or point!=result[index-1])


def _via_path(grid, first, last, via):
    result=[first];visited=set()
    for target in (*via,last):
        leg=grid.path(result[-1],target,forbidden=visited)
        if not leg:
            return []
        visited.update(leg[:-1]);result.extend(leg[1:])
    return result


def _portal_region(goal):
    if isinstance(goal,WallGripGoal):
        return "wall",goal.room,goal.rect,goal.facing
    return goal.kind,goal.key


def _refill_chain_proposals(grid, first, last, anchors, *, max_portals, deadline):
    """Bounded resource chains, without claiming that any dash can reach them.

    Nearby refills on a body-clear directed line form an optional relay. Keep
    progression along the overall journey, rather than permutations that visit
    the same resource or shuttle across an obstacle. A detour may still begin
    away from the destination; ordinary geometric and single-resource routes
    remain separate alternatives.
    """
    if max_portals<2:
        return []
    origin,destination=grid.point(first),grid.point(last)
    direction=(destination[0]-origin[0],destination[1]-origin[1])
    refills=[anchor for anchor in anchors if anchor.kind=="refill"]
    if len(refills)<2 or direction==(0,0):
        return []
    refills=sorted(refills,key=lambda a:(math.dist(origin,(a.x,a.y)),a.key))[:16]
    neighbors={}
    for anchor in refills:
        if time.monotonic()>=deadline:
            return []
        candidates=[]
        for other in refills:
            dx,dy=other.x-anchor.x,other.y-anchor.y
            distance=math.hypot(dx,dy)
            if (12<distance<=128 and dx*direction[0]+dy*direction[1]>0
                    and grid.segment_clear((anchor.x,anchor.y),(other.x,other.y))):
                candidates.append((distance,other.key,other))
        neighbors[anchor.key]=[row[2] for row in sorted(candidates)[:2]]
    frontier=[(anchor,) for anchor in refills[:6]]
    sequences=[]
    def cost(chain):
        points=(origin,*((a.x,a.y) for a in chain),destination)
        return sum(math.dist(a,b) for a,b in zip(points,points[1:]))
    for _ in range(min(4,max_portals)-1):
        if time.monotonic()>=deadline:
            break
        extended=[]
        for chain in frontier:
            for other in neighbors[chain[-1].key]:
                if other not in chain:
                    extended.append((*chain,other))
        frontier=sorted(extended,key=lambda chain:(cost(chain),tuple(a.key for a in chain)))[:8]
        sequences.extend(frontier)
        if not frontier:
            break
    result=[]
    for chain in sorted(sequences,key=lambda chain:(-len(chain),cost(chain),tuple(a.key for a in chain)))[:12]:
        if time.monotonic()>=deadline:
            break
        cells=tuple(grid.nearest((anchor.x,anchor.y)) for anchor in chain)
        if any(cell is None for cell in cells):
            continue
        path=_via_path(grid,first,last,cells)
        if path:
            result.append((path,chain))
    return result


def _resource_route_priority(route, start):
    """Rank relay opportunities, never certify motion or edit resource state.

    A long unbroken airborne span is harder than a similar-distance relay.
    Landing offers a modest jump allowance; an airborne refill does not. Read
    cooldowns from this exact arrival so an immediately reversed route through
    spent refills cannot hide a ready detour. Optimistic travel time permits
    resources that may return before arrival; every pickup is still simulated.
    """
    resources={}
    for portal in route.portals:
        if portal.kind in {"surface","refill"}:
            index=min(range(len(route.points)),key=lambda i:math.dist(
                route.points[i],(portal.x,portal.y)))
            resources.setdefault(index,[]).append(portal)
    cooldowns=dict(start.refill_timers)
    anchor_y=low_y=route.points[0][1]
    jump=40. if start.on_ground and start.vy>=0 else 0.
    travel=span=0.;unready=0;efforts=[]
    for index,point in enumerate(route.points):
        if index:
            distance=math.dist(route.points[index-1],point)
            travel+=distance;span+=distance
        low_y=min(low_y,point[1])
        for portal in resources.get(index,()):
            if portal.kind=="refill":
                remaining=cooldowns.get(getattr(portal,"identity",""),0)
                if remaining>math.ceil(travel/4):
                    unready+=1
                    continue  # Still offer this route; do not count a free refill.
            # A soft distance cost favors continuous relays over spending a
            # whole charge crossing the room before the first refill. It is
            # deliberately not a reachability cutoff for long physical jumps.
            efforts.append(max(0.,anchor_y-low_y-jump)+max(0.,span-96.))
            anchor_y=low_y=point[1];span=0.
            jump=40. if portal.kind=="surface" else 0.
    efforts.append(max(0.,anchor_y-low_y-jump)+max(0.,span-96.))
    return unready,max(efforts),sum(efforts),route.length,route.id


def _interleave(groups):
    """One representative per group before spending slots on its variants."""
    return [group[index] for index in range(max((len(group) for group in groups), default=0))
            for group in groups if index < len(group)]


def _balanced_routes(routes, limit):
    """Keep topology, mechanism conditions, exit lanes and resource roles."""
    topologies = {}
    seen = set()
    for route in routes:
        if route.id in seen:
            continue
        seen.add(route.id)
        winding = tuple(item for item in route.signature
                        if not item.startswith(("via:", "exit_lane:", "requires:")))
        topology = (winding, route.conditional_on)
        lane = tuple(item for item in route.signature if item.startswith("exit_lane:"))
        forced = tuple(item for item in route.signature if item.startswith("via:"))
        role = ("refill" if any(portal.kind == "refill" for portal in route.portals) else
                "wall" if any("wall" in item for item in forced) else
                "support" if forced else "geometric")
        topologies.setdefault(topology, {}).setdefault(lane, {}).setdefault(role, []).append(route)
    groups = []
    for lanes in topologies.values():
        groups.append(_interleave([_interleave(list(roles.values())) for roles in lanes.values()]))
    return _interleave(groups)[:limit]


def _spread_members(members, start):
    """Cover a long opening before revisiting neighboring near-end pixels."""
    remaining = list(members)
    def point(member):
        return member["x"] + 4, member["y"] + 9.5
    selected = [min(remaining, key=lambda member: (math.dist((start.x, start.y), point(member)), point(member)))]
    remaining.remove(selected[0])
    while remaining:
        member = min(remaining, key=lambda row: (-min(math.dist(point(row), point(old)) for old in selected),
                                                 math.dist((start.x, start.y), point(row)), point(row)))
        selected.append(member)
        remaining.remove(member)
    return selected


def _routes_on_grid(simulator, start, goal, grid, *, max_routes, max_portals, deadline,
                    route_filter=None):
    room=grid.room
    approach=_approach_point(room,goal)
    if getattr(goal,"exit_region",()) and not grid.clear(approach):
        # Portal discovery also permits ducking. Its interval midpoint need
        # not fit a standing body, although another pixel in the same lane
        # does. Keep every candidate inside this member's actual bounds.
        bounds=goal.arrival_bounds
        vertical=goal.edge in {"left","right"}
        low,high=(bounds[1]+5.5,bounds[3]+5.5) if vertical else (bounds[0],bounds[2])
        choices=[(approach[0],value) if vertical else (value,approach[1])
                 for value in range(math.ceil(low),math.ceil(high))]
        choices=[point for point in choices if grid.clear(point)]
        if choices:
            approach=min(choices,key=lambda point:(math.dist(point,approach),point))
    first=grid.nearest((start.x,start.y));last=grid.nearest(approach,from_point=False)
    base=grid.path(first,last)
    if not base:
        return []
    anchors=_anchors(simulator,room,grid)
    centers=[(f"solid:{i}",x,y) for i,(x,y) in enumerate(grid.obstacle_centers())]
    # Keep the reference just below the support: standing on its top is a
    # common mandatory portal and must not make the angle signature singular.
    centers += [(f"oneway:{i}",(left+right)/2,top+2)
                for i,(left,top,right,_) in enumerate(grid.platforms)]
    base_points=[grid.point(cell) for cell in base]
    routes=[];signatures=set();structures=set()
    base_length=sum(math.dist(a,b) for a,b in zip(base_points,base_points[1:]))
    def collect(path,forced):
        points=[grid.point(cell) for cell in path]
        length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
        if length>base_length*3.5+128:
            return
        portals=_portals(grid,path,anchors,forced,max_portals)
        if any(anchor not in portals for anchor in forced):
            return
        regions=[]
        for portal in portals:
            region=_portal_region(portal)
            # A real support midway up one wall can divide its ascent. Count
            # that wall once structurally; the directed cell path below still
            # forbids returning through previously traversed geometry.
            if region not in regions:
                regions.append(region)
        simplified=_display_points(grid,points,portals)
        if not simplified:
            return
        # Anchoring a standing support can change how a directed one-way edge
        # is crossed. Classify the final clear polyline, rather than keeping a
        # stale signature for the pre-anchor path (or discarding the resource).
        winding=_signature(simplified,centers)
        signature=winding+tuple('via:'+anchor.key for anchor in forced)
        structure=(winding,tuple(regions))
        if signature in signatures or structure in structures:
            return
        structures.add(structure);signatures.add(signature)
        identifier="corridor:"+hashlib.sha256(repr((room.name,signature)).encode()).hexdigest()[:12]
        length=sum(math.dist(a,b) for a,b in zip(simplified,simplified[1:]))
        candidate=CorridorRoute(identifier,room.name,portals,simplified,signature,length)
        # Conditional geometry must be checked while this slice still has
        # time, not after every optional variant has consumed its budget.
        # This also lets its own cached path checks resume on the next slice.
        if route_filter is not None:
            candidate=route_filter(candidate)
        if candidate is not None:
            routes.append(candidate)
    # A complete clear proposal is useful immediately. Do not wait for all
    # optional detours before giving the physical scheduler its first route.
    collect(base,())
    proposals=[(base,())];penalties={}
    for _ in range(max_routes*2):
        if time.monotonic()>=deadline:
            break
        path=proposals[-1][0]
        for x,y in path:
            for dx in range(-3,4):
                for dy in range(-3,4):
                    cell=(x+dx,y+dy)
                    penalties[cell]=penalties.get(cell,0.)+1.5/(1+max(abs(dx),abs(dy)))
        alternate=grid.path(first,last,penalties,cache_tag=("penalty",_))
        if not alternate:
            break
        proposals.append((alternate,()))
    # Geometry alone cannot know that a detour's support or refill is needed.
    # Retain representatives on both sides instead of ranking only by distance
    # toward the final target. Resource detours are retained separately from
    # geometric bypasses; tiny changes along the same wall are not new routes.
    detours=[]
    for anchor in anchors:
        # A wall can supply another jump even without restoring a dash. Keep
        # real wall contacts as detours, including a lower face beneath an
        # overhang; geometry alone cannot judge the resulting launch. The
        # side of the current launch support is not a useful extra stop.
        if (anchor.kind=="wall_grip" and abs(anchor.x-start.x)<48
                and anchor.y>=start.y-4):
            continue
        point=(anchor.x,anchor.y)
        separation=16 if anchor.kind=="refill" else 24 if anchor.kind=="surface" else 32
        if min(math.dist(point,p) for p in base_points)<=separation:
            continue
        via=grid.nearest(point)
        if via is None:
            continue
        extra=math.dist((start.x,start.y),point)+math.dist(point,approach)
        role=0 if anchor.kind=="refill" else 1 if anchor.kind=="surface" else 2
        side=-1 if anchor.x<start.x else 1
        detours.append((role,extra,side,anchor,via))
    representatives=[];groups=set()
    for row in sorted(detours,key=lambda row:(row[0],row[1],row[3].key)):
        group=(row[0],row[2])
        if group not in groups:
            groups.add(group);representatives.append(row)
    representatives+=sorted(detours,key=lambda row:(row[0],row[1],row[3].key))
    seen_via=set();seen_wall_faces=set()
    for _,_,_,anchor,via in representatives:
        if len(seen_via)>=max_routes*2 or time.monotonic()>=deadline:
            break
        if via in seen_via:
            continue
        wall_face=(anchor.rect,anchor.facing) if isinstance(anchor,WallGripGoal) else None
        if wall_face is not None and wall_face in seen_wall_faces:
            continue  # Heights on one face are physical arrival alternatives.
        seen_via.add(via)
        path=_via_path(grid,first,last,(via,))
        if path:
            proposals.append((path,(anchor,)))
            if wall_face is not None:
                seen_wall_faces.add(wall_face)
    # A refill followed by a nearby higher support is a structurally useful
    # resource chain even when the Euclidean route would immediately turn back.
    # Its actual dash/refill/landing semantics are still left to the solver.
    chains=[]
    for refill in (a for a in anchors if a.kind=="refill"):
        supports=sorted((a for a in anchors if a.kind=="surface" and 4<refill.y-a.y<=96
                         and 12<math.dist((a.x,a.y),(refill.x,refill.y))<=112),
                        key=lambda a:(math.dist((a.x,a.y),(refill.x,refill.y)),a.key))
        for support in supports[:2]:
            if time.monotonic()>=deadline:
                break
            a=grid.nearest((refill.x,refill.y));b=grid.nearest((support.x,support.y))
            path=_via_path(grid,first,last,(a,b))
            if path:
                chains.append((path,(refill,support)))
    # A static wall supplies a new jump, while a later real landing restores
    # resources for the final ascent. Preserve the pair: a straight geometric
    # suffix from the wall can otherwise omit the only useful landing. These
    # are proposals only; the solver must reach both contacts continuously.
    wall_chains=[];chain_faces=set()
    for _,_,_,wall,_ in sorted(detours,key=lambda row:(row[0],row[1],row[3].key)):
        if not isinstance(wall,WallGripGoal) or (wall.rect,wall.facing) in chain_faces:
            continue
        chain_faces.add((wall.rect,wall.facing))
        supports=sorted((a for a in anchors if a.kind=="surface"
                         and 16<math.dist((wall.x,wall.y),(a.x,a.y))<=208
                         and abs(a.y-wall.y)<=96),
            key=lambda a:(math.dist((wall.x,wall.y),(a.x,a.y))
                          + .25*math.dist((a.x,a.y),approach),a.key))
        for support in supports[:2]:
            if time.monotonic()>=deadline:
                break
            path=_via_path(grid,first,last,(grid.nearest((wall.x,wall.y)),
                                           grid.nearest((support.x,support.y))))
            if path:
                wall_chains.append((path,(wall,support)))
        if len(wall_chains)>=max_routes or time.monotonic()>=deadline:
            break
    refill_chains=_refill_chain_proposals(grid,first,last,anchors,
        max_portals=max_portals,deadline=deadline)
    # Give each obstacle bypass a resource-chain representative before more
    # variants of one side. Longer coherent relays precede their short prefixes;
    # lower and upper detours still compete within the same total route cap.
    relay_representatives=[];relay_remainder=[];relay_windings=set()
    for proposal in refill_chains:
        winding=_signature([grid.point(cell) for cell in proposal[0]],centers)
        if winding in relay_windings:
            relay_remainder.append(proposal)
        else:
            relay_windings.add(winding);relay_representatives.append(proposal)
    # Preserve genuinely different obstacle bypasses before optional resource
    # chains. A penalized path with the same winding is not another corridor.
    geometric=[];windings={_signature(base_points,centers)}
    for path,forced in proposals[1:]:
        winding=_signature([grid.point(cell) for cell in path],centers)
        if not forced and winding not in windings:
            windings.add(winding);geometric.append((path,forced))
    proposals=([proposals[0]]+geometric[:1]+relay_representatives+chains+wall_chains+
               [row for row in proposals[1:] if row[1]]+relay_remainder+geometric[1:])
    for path,forced in proposals:
        collect(path,forced)
        if len(routes)>=max(24,max_routes*4) or time.monotonic()>=deadline:
            break
    routes=_balanced_routes(routes,max_routes)
    return routes


def _current_grid(simulator, room, state, step, deadline, omitted=frozenset()):
    """Per-call occupancy; moving geometry never enters the static LRU."""
    mechanisms={key:(entity,rect) for key,entity,rect in simulator._mechanism_rects(room.name,state)}
    hazards=[]
    attachments=simulator._spike_attachments[room.name]
    for hazard in simulator._hazards[room.name]:
        rect=_hazard_rect(hazard)
        owner=attachments.get((hazard.kind,hazard.id))
        if owner is not None:
            identity=f"{room.name}:{owner.id}"
            if identity in omitted or identity not in mechanisms:
                continue
            _,bounds=mechanisms[identity]
            dx,dy=bounds[0]-owner.x,bounds[1]-owner.y
            rect=tuple(value+(dx if index%2==0 else dy) for index,value in enumerate(rect))
        hazards.append(rect)
    return _ClearanceGrid(simulator,room,step,deadline,
        extra_solids=[rect for key,(_,rect) in mechanisms.items() if key not in omitted],
        hazard_rects=hazards)


def _path_progress(points, point):
    distances=[math.dist(point,old) for old in points]
    index=min(range(len(points)),key=lambda i:distances[i])
    return sum(math.dist(a,b) for a,b in zip(points[:index],points[1:index+1]))


def _gate_intersections(points, entries):
    """Find occupied portions of a proposed path, retaining required body space."""
    hits={};travel=0.
    for before,after in zip(points,points[1:]):
        distance=math.dist(before,after);count=max(1,math.ceil(distance/2))
        for tick in range(count+1):
            x=before[0]+(after[0]-before[0])*tick/count
            y=before[1]+(after[1]-before[1])*tick/count
            body=(x-4,y-11,x+4,y)
            for identity,(_,rect,hazards) in entries.items():
                if not any(_overlap(body,box) for box in (rect,*hazards)):
                    continue
                progress=travel+distance*tick/count
                if identity not in hits:
                    hits[identity]=[progress,body,(x,y)]
                else:
                    old=hits[identity][1]
                    hits[identity][1]=(min(old[0],body[0]),min(old[1],body[1]),
                                       max(old[2],body[2]),max(old[3],body[3]))
        travel+=distance
    return sorted(((value[0],key,value[1],value[2]) for key,value in hits.items()))


def _trigger_grip(grid, origin, identity, entity, rect, contact):
    """Choose a real, accessible side of the occupied block, not its centre."""
    left,top,right,bottom=rect
    heights={min(bottom+8,max(top+4,contact[1])),bottom+8,top+12}
    candidates=[]
    original=(entity.x,entity.y,entity.x+entity.width,entity.y+entity.height)
    for facing,x in ((1,left-4),(-1,right+4)):
        for y in heights:
            if not top<y<bottom+11 or not grid.clear((x,y)):
                continue
            candidates.append((math.dist((x,y),contact),x,y,facing))
    first=grid.nearest(origin)
    for _,x,y,facing in sorted(candidates):
        target=grid.nearest((x,y),from_point=False)
        if grid.path(first,target):
            trigger_type=ZipMoverTriggerGoal if entity.kind=="zipMover" else FallingBlockTriggerGoal
            prefix="zip_trigger" if entity.kind=="zipMover" else "falling_trigger"
            return trigger_type("wall_grip",grid.room.name,x,y,
                id=f"{prefix}:{identity}:{facing}",rect=original,
                facing=facing,identity=identity,mechanism_kind=entity.kind,
                grip_offset=y-top)
    return None


def _clear_patch(grid, x, y):
    boxes=[(x-2,y-2,x+2,y+2)]
    boxes.extend((x+dx,y+dy,x+dx+4,y+dy+4) for dx in (-4,0) for dy in (-4,0))
    valid=[box for box in boxes if all(grid.clear((px,py))
           for px in range(math.floor(box[0]),math.ceil(box[2])+1)
           for py in range(math.floor(box[1]),math.ceil(box[3])+1))]
    return min(valid,key=lambda box:math.dist(((box[0]+box[2])/2,(box[1]+box[3])/2),(x,y))) \
        if valid else (x,y,x,y)


def _dash_approach(grid, origin, identity, rect, contact):
    """An accessible cardinal impact face, with attached hazards still solid.

    Clearance up to the collision boundary excludes a spiked face. These are
    geometric approaches only: the later physical edge must actually spend a
    dash, hit this block, and carry the resulting rebound state onward.
    """
    left,top,right,bottom=rect
    heights={min(bottom+8,max(top+4,contact[1])),top+12,(top+bottom+11)/2,bottom+8}
    widths={min(right-4,max(left+4,contact[0])),left+4,(left+right)/2,right-4}
    candidates=[]
    for y in heights:
        if top<y<bottom+11:
            candidates.extend(((left-8,y,1,0,(left-4,y)),
                               (right+8,y,-1,0,(right+4,y))))
    for x in widths:
        if left<x<right:
            candidates.extend(((x,top-4,0,1,(x,top)),
                               (x,bottom+15,0,-1,(x,bottom+11))))
    first=grid.nearest(origin)
    candidates.sort(key=lambda row:(math.dist(row[:2],contact)+.1*math.dist(row[:2],origin),row[:4]))
    for x,y,dx,dy,impact in candidates:
        if not grid.clear((x,y)) or not grid.segment_clear((x,y),impact):
            continue
        target=grid.nearest((x,y),from_point=False)
        if grid.path(first,target):
            approach=DashBlockApproachGoal("dash_approach",grid.room.name,x,y,
                id=f"dash_approach:{identity}:{dx}:{dy}",identity=identity,
                bounds=_clear_patch(grid,x,y),role="passage")
            return approach,dx,dy
    return None


def _passage_goal(grid, before, after, rect, clearance):
    """Choose a clear bend inside the newly available mechanism footprint."""
    path=grid.path(grid.nearest(before),grid.nearest(after,from_point=False))
    if not path:
        return None
    points=[grid.point(cell) for cell in path]
    def inside(point):
        x,y=point
        return _overlap((x-4,y-11,x+4,y),rect)
    occupied=[point for point in points if inside(point)]
    if not occupied:
        occupied=[point for point in points if _overlap(
            (point[0]-4,point[1]-11,point[0]+4,point[1]),clearance.clearance_box)]
    if not occupied:
        return None
    bends=[point for point in _simplify(grid,points)[1:-1] if point in occupied]
    x,y=bends[-1] if bends else occupied[len(occupied)//2]
    # Keep an actual body-clear acceptance patch. Near a ceiling corner, a
    # symmetric radius would include solid space; use the clear quadrant.
    boxes=[(x-2,y-2,x+2,y+2)]
    boxes.extend((x+dx,y+dy,x+dx+4,y+dy+4) for dx in (-4,0) for dy in (-4,0))
    valid=[box for box in boxes if all(grid.clear((px,py))
           for px in range(int(box[0]),int(box[2])+1)
           for py in range(int(box[1]),int(box[3])+1))]
    bounds=min(valid,key=lambda box:math.dist(((box[0]+box[2])/2,(box[1]+box[3])/2),after)) \
        if valid else (x,y,x,y)
    if isinstance(clearance,MechanismClearanceGoal) and clearance.mechanism_kind=="zipMover":
        dx,dy=rect[0]-clearance.rect[0],rect[1]-clearance.rect[1]
        footprints=(tuple(rect),)+tuple((a+dx,b+dy,c+dx,d+dy)
            for a,b,c,d in clearance.attached_hazards)
        indices=[index for index,point in enumerate(points) if any(_overlap(
            (point[0]-4,point[1]-11,point[0]+4,point[1]),box) for box in footprints)]
        if not indices:
            return None
        last=indices[-1]
        following=points[last+1] if last+1<len(points) else after
        direction=(following[0]-points[last][0],following[1]-points[last][1])
        if direction==(0,0):
            direction=(after[0]-before[0],after[1]-before[1])
        if direction==(0,0):
            return None
        axis=0 if abs(direction[0])>=abs(direction[1]) else 1
        sign=1 if direction[axis]>0 else -1
        return ZipMoverPassageGoal("corridor",grid.room.name,x,y,
            id=f"mechanism_passage:{clearance.identity}",bounds=bounds,identity=clearance.identity,
            occupied_rect=tuple(rect),footprints=footprints,crossing_axis=axis,
            crossing_sign=sign,clearance=clearance)
    passage_type=DashBlockPassageGoal if isinstance(clearance,DashBlockBreakGoal) else MechanismPassageGoal
    return passage_type("corridor",grid.room.name,x,y,
        id=f"mechanism_passage:{clearance.identity}",bounds=bounds,identity=clearance.identity)


def _conditional_route(simulator, route, entries, state, grid_for, deadline):
    intersections=_gate_intersections(route.points,entries)
    if not intersections:
        return None
    timeline=[(_path_progress(route.points,(portal.x,portal.y)),2,index,portal)
              for index,portal in enumerate(route.portals)]
    omitted=set();origin=route.points[0]
    records={row[0]:row for row in state.mechanisms}
    for index,(progress,identity,clearance,contact) in enumerate(intersections):
        if time.monotonic()>=deadline:
            return None
        entity,rect,hazards=entries[identity]
        if entity.kind=="dashBlock":
            entry=_dash_approach(grid_for(frozenset(omitted)),origin,identity,rect,contact)
            if entry is None:
                return None
            approach,dx,dy=entry
            timeline.append((progress-.01,0,index,approach))
            clearance_goal=DashBlockBreakGoal("dash_block_break",route.room,approach.x,approach.y,
                id=f"dash_break:{identity}",identity=identity,rect=rect,
                clearance_box=clearance,dash_dx=dx,dash_dy=dy)
            timeline.append((progress-.01,1,index,clearance_goal))
            omitted.add(identity);origin=(approach.x,approach.y)
            continue
        grip=_trigger_grip(grid_for(frozenset(omitted)),origin,identity,entity,rect,contact)
        if grip is None:
            return None
        phase=records.get(identity,(None,0))[1]
        if phase==0:
            timeline.append((progress-.01,0,index,grip))
        clearance_goal=MechanismClearanceGoal("mechanism_clearance",route.room,grip.x,grip.y,
            id=f"mechanism_clearance:{identity}",identity=identity,rect=grip.rect,
            clearance_box=clearance,facing=grip.facing,mechanism_kind=entity.kind,
            attached_hazards=tuple(_hazard_rect(hazard) for hazard in
                simulator._hazards[route.room]
                if simulator._spike_attachments[route.room].get((hazard.kind,hazard.id))==entity))
        timeline.append((progress-.01,1,index,clearance_goal))
        omitted.add(identity);origin=(grip.x,grip.y)
    ordered=tuple(row[3] for row in sorted(timeline,key=lambda row:row[:3]))
    expanded=[];omitted=set();previous=route.points[0]
    for index,portal in enumerate(ordered):
        expanded.append(portal)
        if isinstance(portal,(MechanismClearanceGoal,DashBlockBreakGoal)):
            omitted.add(portal.identity)
            following=next((candidate for candidate in ordered[index+1:]
                            if not isinstance(candidate,(MechanismClearanceGoal,DashBlockBreakGoal))),None)
            after=route.points[-1] if following is None else (following.x,following.y)
            passage=_passage_goal(grid_for(frozenset(omitted)),previous,after,
                                  entries[portal.identity][1],portal)
            if passage is None:
                return None
            expanded.append(passage);previous=(passage.x,passage.y)
        else:
            previous=(portal.x,portal.y)
    portals=tuple(expanded)
    # Each approach is routed against the still-occupied blocks. Only after an
    # explicit clearance condition may preview geometry pass their old pose.
    # The physical solver still checks their actual new pose on every frame.
    points=[route.points[0]];omitted=set()
    for portal in (*portals,None):
        if isinstance(portal,(MechanismClearanceGoal,DashBlockBreakGoal)):
            omitted.add(portal.identity)
            continue
        point=route.points[-1] if portal is None else (portal.x,portal.y)
        grid=grid_for(frozenset(omitted))
        path=grid.path(grid.nearest(points[-1]),grid.nearest(point,from_point=False))
        if not path:
            return None
        piece=_simplify(grid,[grid.point(cell) for cell in path])
        for next_point in (*piece,point):
            if points[-1]==next_point:
                continue
            if not grid.segment_clear(points[-1],next_point):
                return None
            points.append(next_point)
    # Reconnection through the actual grip can change the precise crossing.
    # Require the union of both footprints, not just the optimistic draft's.
    final_boxes={row[1]:row[2] for row in _gate_intersections(points,entries)}
    revised=[]
    for portal in portals:
        if isinstance(portal,(MechanismClearanceGoal,DashBlockBreakGoal)) and portal.identity in final_boxes:
            a,b,c,d=portal.clearance_box;e,f,g,h=final_boxes[portal.identity]
            portal=replace(portal,clearance_box=(min(a,e),min(b,f),max(c,g),max(d,h)))
        revised.append(portal)
    clearances={portal.identity:portal for portal in revised if isinstance(portal,MechanismClearanceGoal)}
    portals=tuple(replace(portal,clearance=clearances[portal.identity])
                  if isinstance(portal,ZipMoverPassageGoal) else portal for portal in revised)
    dependencies=tuple(row[1] for row in intersections)
    signature=route.signature+tuple(f"requires:{identity}" for identity in dependencies)
    identity="corridor:"+hashlib.sha256(repr((route.room,signature)).encode()).hexdigest()[:12]
    length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
    return CorridorRoute(identity,route.room,portals,tuple(points),signature,length,dependencies)


def _conditional_entries(simulator, room, start):
    """Only modeled, changeable occupants can open a conditional passage."""
    records={row[0]:row for row in start.mechanisms}
    entries={}
    for identity,entity,rect in simulator._mechanism_rects(room.name,start):
        phase=records.get(identity,(None,0))[1]
        falling=(entity.kind=="fallingBlock" and phase!=3
                 and (phase!=0 or _truth(entity.attrs.get("climbFall",True))))
        dash_break=(entity.kind=="dashBlock" and _truth(entity.attrs.get("canDash",True)))
        zip_move=(entity.kind=="zipMover" and entity.nodes
                  and entity.nodes[0]!=(entity.x,entity.y))
        if not (falling or dash_break or zip_move):
            continue
        hazards=[]
        for hazard in simulator._hazards[room.name]:
            if simulator._spike_attachments[room.name].get((hazard.kind,hazard.id))==entity:
                a,b,c,d=_hazard_rect(hazard);dx,dy=rect[0]-entity.x,rect[1]-entity.y
                hazards.append((a+dx,b+dy,c+dx,d+dy))
        entries[identity]=(entity,rect,tuple(hazards))
    return entries


def _discovery_scenarios(entries, room, goal):
    destination=_approach_point(room,goal)
    ranked=sorted(entries,key=lambda key:(math.dist(
        ((entries[key][1][0]+entries[key][1][2])/2,
         (entries[key][1][1]+entries[key][1][3])/2),destination),key))
    yield frozenset()
    if not ranked:
        return
    # The all-gate proposal must remain available even with more than six
    # required gates. Its actual dependency set is still proved sequentially.
    yield frozenset((ranked[0],))
    if len(ranked)>1:
        yield frozenset(ranked)
    for key in ranked[1:]:
        yield frozenset((key,))
    # Lazy diagonals visit pairs involving different gates before spending
    # more work beside the nearest one. Never materialize n-choose-2 tasks.
    for gap in range(1,len(ranked)):
        for index in range(len(ranked)-gap):
            if len(ranked)>2:
                yield frozenset((ranked[index],ranked[index+gap]))


class _CorridorDiscovery:
    """Persist completed lane/scenario work across short discovery slices."""
    def __init__(self, simulator, start, goal, cell_size, max_portals):
        self.simulator,self.start,self.goal=simulator,start,goal
        self.cell_size,self.max_portals=cell_size,max_portals
        self.data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
        self.pool=[];self.raw=[];self.known=set();self.grids={};self.cursor=0
        self.jobs=[];self.retries=deque();self.attempted=0
        if (start.dead or start.room not in self.data.rooms
                or (getattr(goal,"source_room","") or goal.room)!=start.room):
            return
        room=self.data.rooms[start.room]
        self.entries=_conditional_entries(simulator,room,start)
        members=getattr(goal,"region_members",lambda:())()
        lanes=[]
        if getattr(goal,"exit_region",()) and members:
            for member in _spread_members(members,start)[:_CANDIDATE_POOL_LIMIT]:
                left,top,right,bottom=member["arrival_bounds"]
                lane=replace(goal,tile_x=member["x"],tile_y=member["y"],
                    x=(left+right)/2,y=(top+bottom)/2+5.5,
                    arrival_bounds=tuple(member["arrival_bounds"]))
                lanes.append((lane,(member["x"],member["y"],tuple(member["arrival_bounds"]))))
        else:
            lanes=[(goal,None)]
        for lane_index,(lane,key) in enumerate(lanes):
            scenarios=_discovery_scenarios(self.entries,room,lane)
            first=next(scenarios)
            job=dict(lane=lane,key=key,scenario=first,attempts=0)
            heapq.heappush(self.jobs,(lane_index,0,lane_index,job,scenarios))

    def next_job(self):
        if self.retries and (not self.jobs or len(self.retries)>=4 or self.attempted%4==3):
            return self.retries.popleft()
        _,index,lane_index,job,scenarios=heapq.heappop(self.jobs)
        following=next(scenarios,None)
        if following is not None:
            pending=dict(lane=job["lane"],key=job["key"],scenario=following,attempts=0)
            heapq.heappush(self.jobs,(lane_index+index+1,index+1,lane_index,pending,scenarios))
        return job

    @property
    def complete(self):
        return not (self.jobs or self.retries) or len(self.pool)>=_CANDIDATE_POOL_LIMIT

    @property
    def conditional_work_pending(self):
        if not getattr(self,"entries",None):
            return False
        # A lane's initial ordinary head still owns an unconsumed scenario
        # iterator. Later heads are all conditional subsets. A timed-out
        # ordinary retry alone does not imply more conditional work exists.
        return (any(index==0 or job["scenario"] for _,index,_,job,_ in self.jobs)
                or any(job["scenario"] for job in self.retries))

    def advance(self, deadline):
        # A failed time slice is rotated behind unseen tasks. Completed grids
        # and condition subsets remain cached; they are never restarted just
        # to rediscover the same first six routes on a later page.
        while (self.jobs or self.retries) and time.monotonic()<deadline and len(self.raw)<_CANDIDATE_POOL_LIMIT*4:
            job=self.next_job()
            self.attempted+=1
            now=time.monotonic()
            share=(deadline-now)/max(1,min(4,len(self.jobs)+len(self.retries)+1))
            share=min(deadline-now,share*(2**min(job["attempts"],3)))
            local_deadline=now+share
            routes=_generate_corridor_target(self.simulator,self.start,job["lane"],
                max_routes=_CANDIDATE_POOL_LIMIT if job["key"] is None and not job["scenario"] else 6,
                cell_size=self.cell_size,max_portals=self.max_portals,deadline=local_deadline,
                discovery_scenario=job["scenario"],discovery_entries=self.entries,
                discovery_grids=self.grids)
            if time.monotonic()>=local_deadline:
                job["attempts"]+=1
                self.retries.append(job)
            else:
                self.cursor+=1
            # Caller-owned continuations can outlive the small global LRU.
            # Keep only a few grids useful to unfinished tasks, releasing the
            # potentially large geometry of every completed condition subset.
            retry_scenarios={pending["scenario"] for pending in self.retries}
            unfinished=[(key,grid) for key,grid in self.grids.items()
                        if not grid.complete or any(not row["done"] for row in grid._paths.values())]
            critical={key for key,_ in unfinished}
            useful=[(key,grid) for key,grid in self.grids.items()
                    if key not in critical and (key in retry_scenarios or not key and retry_scenarios)]
            self.grids.clear()
            self.grids.update(unfinished)
            self.grids.update(useful[-max(0,8-len(unfinished)):] if len(unfinished)<8 else [])
            for route in routes:
                if job["key"] is not None:
                    signature=route.signature+("exit_lane:"+repr(job["key"]),)
                    identifier="corridor:"+hashlib.sha256(repr((route.room,signature)).encode()).hexdigest()[:12]
                    route=replace(route,id=identifier,signature=signature)
                if route.id not in self.known:
                    # Preserve discovery space for conditional candidates as
                    # well as published slots. Otherwise 96 ordinary raw
                    # variants can stop all work with reserved slots empty.
                    if (self.entries and not route.conditional_on
                            and sum(not row.conditional_on for row in self.raw)>=_CANDIDATE_POOL_LIMIT*3):
                        continue
                    if len(self.raw)>=_CANDIDATE_POOL_LIMIT*4:
                        break
                    self.known.add(route.id);self.raw.append(route)
            if time.monotonic()>=local_deadline and len(self.jobs)+len(self.retries)==1:
                break
        published={route.id for route in self.pool}
        available=[route for route in self.raw if route.id not in published]
        slots=_CANDIDATE_POOL_LIMIT-len(self.pool)
        missing=max(0,_CONDITIONAL_RESERVE-sum(bool(route.conditional_on) for route in self.pool))
        reserved=(missing if self.conditional_work_pending else
                  min(missing,sum(bool(route.conditional_on) for route in available)))
        ordinary_slots=max(0,slots-reserved)
        additions=[]
        for route in _balanced_routes(available,len(available)):
            if not route.conditional_on:
                if ordinary_slots<=0:
                    continue
                ordinary_slots-=1
            additions.append(route)
            if len(additions)>=slots:
                break
        room=self.data.rooms.get(self.start.room)
        if room is not None and sum(entity.kind=="refill" for entity in room.entities)>=2:
            additions=[route for index in range(0,len(additions),6)
                       for route in sorted(additions[index:index+6],
                                           key=lambda item:_resource_route_priority(item,self.start))]
        self.pool.extend(additions)


def generate_corridor_routes(simulator, start, goal, *, max_routes=6, cell_size=4,
                             max_portals=10, deadline=None, candidate_offset=0,
                             continuation=None):
    """Return a page from a bounded, state-specific geometric inventory.

    A page is not the full search space. Later pages preserve other discovered
    topologies instead of generating the same nearest six routes repeatedly.
    Cached proposals are geometry only: full physical replay remains mandatory.
    A caller may retain ``page.continuation`` while its stage is paused, then
    pass it back to preserve discovery even after the optional LRU evicts it.
    """
    if (cell_size not in (2,4,8) or max_routes<1 or max_portals<1
            or type(candidate_offset) is not int or candidate_offset<0):
        raise ValueError("invalid corridor grid, candidate limit or offset")
    deadline=time.monotonic()+2 if deadline is None else deadline
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    key=(id(data),start,repr(goal),cell_size,max_portals,getattr(simulator,"max_dashes",1))
    if continuation is not None:
        if (not isinstance(continuation,_CorridorDiscovery) or continuation.data is not data
                or getattr(continuation,"cache_key",None)!=key):
            raise ValueError("corridor continuation does not match the complete start, goal or map")
        discovery=continuation
    else:
        discovery=_CANDIDATE_CACHE.get(key)
        if discovery is not None and discovery.data is data:
            _CANDIDATE_CACHE.move_to_end(key)
        else:
            discovery=None
    if candidate_offset>=_CANDIDATE_POOL_LIMIT:
        return CorridorCandidates(complete=True,reason="candidate_pool_limit",next_offset=candidate_offset,
                                  continuation=discovery)
    if time.monotonic()>=deadline:
        return CorridorCandidates(next_offset=candidate_offset,continuation=discovery)
    if discovery is None:
        discovery=_CorridorDiscovery(simulator,start,goal,cell_size,max_portals)
        discovery.cache_key=key
        _CANDIDATE_CACHE[key]=discovery
        while len(_CANDIDATE_CACHE)>_GRID_CACHE_LIMIT:
            _CANDIDATE_CACHE.popitem(last=False)
    before=time.monotonic()
    if candidate_offset>=len(discovery.pool) and not discovery.complete:
        discovery.advance(min(deadline,before+2))
    pool=discovery.pool
    page=pool[candidate_offset:candidate_offset+max_routes]
    reason=("candidate_pool_limit" if len(pool)>=_CANDIDATE_POOL_LIMIT else
            "bounded_inventory_complete" if discovery.complete else "generation_time_budget")
    return CorridorCandidates(page,complete=discovery.complete,reason=reason,
        next_offset=candidate_offset+len(page),pool_size=len(pool),
        generation_seconds=time.monotonic()-before,discovery_cursor=discovery.cursor,
        continuation=discovery)



def _generate_corridor_target(simulator, start, goal, *, max_routes=6, cell_size=4,
                             max_portals=10, deadline=None, discovery_scenario=None,
                             discovery_entries=None, discovery_grids=None):
    """One resumable discovery task: a fixed opening and mechanism subset.

    Current occupancy and every still-closed mechanism remain in the grid.
    Conditional proposals must prove the real trigger, clearance and passage
    sequence before becoming candidates; they are never physical successes.
    """
    if cell_size not in (2,4,8) or max_routes<1 or max_portals<1:
        raise ValueError("invalid corridor grid or candidate limit")
    deadline=time.monotonic()+2 if deadline is None else deadline
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    if (start.dead or start.room not in data.rooms or time.monotonic()>=deadline
            or (getattr(goal,"source_room","") or goal.room)!=start.room):
        return []
    room=data.rooms[start.room]
    has_dynamic=bool(simulator._mechanisms[room.name] or simulator._cassette_entities[room.name])
    grids={} if discovery_grids is None else discovery_grids
    if not has_dynamic:
        grid=grids.get(frozenset())
        if grid is None:
            grid=_clearance_grid(simulator,room,cell_size,deadline)
            grids[frozenset()]=grid
        else:
            grid.resume_build(deadline)
        return _routes_on_grid(simulator,start,goal,grid,max_routes=max_routes,
                               max_portals=max_portals,deadline=deadline)
    def grid_for(omitted):
        if omitted not in grids:
            grid=_current_grid(simulator,room,start,cell_size,deadline,omitted)
            grids[omitted]=grid
            return grid
        grids[omitted].resume_build(deadline)
        return grids[omitted]
    scenario=frozenset() if discovery_scenario is None else discovery_scenario
    entries=_conditional_entries(simulator,room,start) if discovery_entries is None else discovery_entries
    selected={key:entries[key] for key in scenario}
    route_filter=(lambda route:_conditional_route(simulator,route,selected,start,grid_for,deadline)) \
        if scenario else None
    return _routes_on_grid(simulator,start,goal,grid_for(scenario),
        max_routes=max_routes,max_portals=max_portals,deadline=deadline,route_filter=route_filter)
