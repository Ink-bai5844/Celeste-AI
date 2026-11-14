"""Request-local fixed-terrain progress rewards, used only to order exploration.

This module never steps physics, returns an executable edge, changes a goal,
or rejects a branch. Missing, ambiguous and unreachable geometry contributes
zero cost so the existing capability heuristic remains available.
"""
from __future__ import annotations

from collections import OrderedDict
import heapq
import math
import time

from .corridor_routes import _ClearanceGrid, _approach_point, _hazard_rect
from .physics import MECHANISM_KINDS, _spike_rides

_DIRECTIONS = ((0,-1),(-1,0),(1,0),(0,1),(-1,-1),(1,-1),(-1,1),(1,1))


class StaticDetourGuide:
    """Bounded reverse distance fields for one search request and map.

    Add ``score(state, public_goal)`` to the existing exploration priority.
    Call ``prepare(start, goal)`` at a public segment's actual starting state.
    For a terrain-blocked segment, known distance-field progress receives a
    negative reward relative to that reference. Unknown geometry stays zero.
    Movers and their attached hazards are absent from these optimistic fields.
    """

    def __init__(self, simulator, *, deadline, cell_size=8, build_seconds=.75):
        self.simulator = simulator
        self.deadline = deadline
        self.cell_size = cell_size if cell_size in (4,8) else 8
        # Building a field once can take longer on a busy/E-core worker than
        # in an isolated run. Do not silently lose all geometry guidance due
        # to a tiny setup slice; the request deadline and cell cap still bind.
        self.build_seconds = max(0., min(.75, build_seconds))
        self._fields = OrderedDict()
        self._references = OrderedDict()
        self._rooms = set()
        self._queries = OrderedDict()
        self.builds = self.failures = self.hits = 0

    def _field(self, state, goal):
        data = getattr(self.simulator, "map", None)
        rooms = getattr(data, "rooms", {})
        source_room = getattr(goal, "source_room", "") or goal.room
        if (state.room != source_room or state.room not in rooms
                or not all(hasattr(self.simulator,name) for name in
                           ("_static_rects","_platforms","_hazards"))):
            return None
        room = rooms[state.room]
        point = _approach_point(room, goal)
        key = (state.room, point)
        if key in self._fields:
            self.hits += 1
            self._fields.move_to_end(key)
            return self._fields[key]
        if state.room not in self._rooms and len(self._rooms) >= 3:
            return None
        self._rooms.add(state.room)
        self.builds += 1
        end = min(self.deadline, time.monotonic()+self.build_seconds)
        result = None
        nx = int((room.width-8)//self.cell_size)+1
        ny = int((room.height-12)//self.cell_size)+1
        if time.monotonic() < end and nx*ny <= 12000:
            movers = [e for e in room.entities
                      if e.kind in MECHANISM_KINDS or e.kind == "cassetteBlock"]
            hazards = [_hazard_rect(e) for e in self.simulator._hazards[state.room]
                       if not any(_spike_rides(e,mover) for mover in movers)]
            grid = _ClearanceGrid(self.simulator,room,self.cell_size,end,hazard_rects=hazards)
            last = grid.nearest(point,from_point=False) if grid.complete else None
            if last is not None:
                distances = {last:math.dist(grid.point(last),point)}
                queue = [(distances[last],last)]
                while queue and time.monotonic() < end:
                    distance,cell = heapq.heappop(queue)
                    if distance != distances[cell]:
                        continue
                    x,y = cell
                    for dx,dy in _DIRECTIONS:
                        previous = (x+dx,y+dy)
                        if previous not in grid.free:
                            continue
                        if dx and dy and ((x+dx,y) not in grid.free or (x,y+dy) not in grid.free):
                            continue
                        # Reverse the directed edge: downward jump-through
                        # crossings differ from upward crossings.
                        if not grid.segment_clear(grid.point(previous),grid.point(cell)):
                            continue
                        value = distance+self.cell_size*math.hypot(dx,dy)
                        if value < distances.get(previous,math.inf):
                            distances[previous] = value
                            heapq.heappush(queue,(value,previous))
                if not queue:
                    result = (grid,distances,point)
        if result is None:
            self.failures += 1
        self._fields[key] = result
        while len(self._fields) > 4:
            self._fields.popitem(last=False)
        return result

    def _distance(self, state, field):
        grid,distances,target = field
        point = (state.x,state.y)
        key = (state.room,target,point)
        if key in self._queries:
            self._queries.move_to_end(key)
            return self._queries[key]
        value = None
        cell = grid.nearest(point)
        if cell in distances:
            # A short clear connector is mandatory. Do not jump to the
            # other side of a wall to manufacture a finite distance.
            connector = math.dist(point,grid.point(cell))
            value = distances[cell]+connector
            # Equidistant projections with incompatible connectivity are a
            # coarse-grid ambiguity, so cannot establish known progress.
            for dx,dy in _DIRECTIONS:
                other = (cell[0]+dx,cell[1]+dy)
                if (other in grid.free
                        and abs(math.dist(point,grid.point(other))-connector)<1e-7
                        and grid.segment_clear(point,grid.point(other))):
                    if (other not in distances or
                            abs(distances[other]-distances[cell])>grid.step*math.sqrt(2)+1e-7):
                        value = None
                        break
                    value = min(value,distances[other]+connector)
        self._queries[key] = value
        while len(self._queries) > 1024:
            self._queries.popitem(last=False)
        return value

    def prepare(self, start, goal):
        """Anchor (or explicitly re-anchor) this segment to a real start.

        Open straight corridors are left unchanged. A lazy first score also
        prepares an unseen field, but callers should prepare the original
        public-segment start before scoring alternative arrivals.
        """
        if start.dead or time.monotonic() >= self.deadline:
            return False
        field = self._field(start,goal)
        if field is None:
            return False
        grid,_,target = field
        reference = (None if grid.segment_clear((start.x,start.y),target)
                     else self._distance(start,field))
        key = (start.room,target)
        self._references[key] = reference
        self._references.move_to_end(key)
        while len(self._references) > 4:
            self._references.popitem(last=False)
        return reference is not None

    def score(self, state, goal):
        if state.dead or time.monotonic() >= self.deadline:
            return 0.
        field = self._field(state,goal)
        if field is None:
            return 0.
        key = (state.room,field[2])
        if key not in self._references:
            self.prepare(state,goal)
        reference = self._references.get(key)
        if reference is None:
            return 0.
        self._references.move_to_end(key)
        distance = self._distance(state,field)
        return 0. if distance is None else .25*min(0.,distance-reference)

    def cache_info(self):
        return dict(fields=len(self._fields),references=len(self._references),
                    rooms=len(self._rooms),queries=len(self._queries),
                    builds=self.builds,failures=self.failures,hits=self.hits,
                    max_fields=4,max_references=4,max_rooms=3,max_queries=1024,
                    build_seconds=self.build_seconds,cell_size=self.cell_size)
