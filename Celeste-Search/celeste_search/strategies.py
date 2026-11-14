"""Geometric subgoals for offline route search; all edges use real simulation.

High berries above springs are easier to search as two connected problems:
activate the spring, then collect and land. No player state is manufactured or
teleported. This module uses only the project's offline physics/search code.
"""

from __future__ import annotations

from dataclasses import dataclass,replace
import math
import time

from .planner import Goal, Heuristic, SearchResult, berry_id, coverage_goals, search_leg, winged_berries
from .physics import spring_player_enabled


class SpringGoal(Goal):
    """Reach the simulator's actual spring-launch event, not mere proximity."""

    def achieved(self, state) -> bool:
        return (
            not state.dead and state.room == self.room
            and getattr(state, "auto_jump_active", False) and state.vy <= -180
            and abs(state.x - self.x) < 12 and abs(state.y - (self.y-6)) <= 1
        )


@dataclass(frozen=True)
class WallSpringGoal(Goal):
    direction: int=1

    def achieved(self,state):
        return (not state.dead and state.room==self.room and state.auto_jump_active
                and abs(state.vx-self.direction*240)<.01 and abs(state.vy+140)<.01
                and state.force_x==self.direction and state.force_frames==18
                and abs(state.x-(self.x+self.direction*10))<.01
                and abs(state.y-self.y)<16)


def usable_spring(entity):
    return (entity.kind in {"spring","wallSpringLeft","wallSpringRight"}
            and spring_player_enabled(entity) and entity.attrs.get("collidable") is not False)


def spring_goal(room_name,entity):
    if not usable_spring(entity):
        return None
    identity=f"spring:{room_name}:{entity.id}"
    if entity.kind=="spring":
        return SpringGoal("spring",room_name,entity.x,entity.y,id=identity)
    return WallSpringGoal("wall_spring",room_name,entity.x,entity.y,id=identity,
                          direction=1 if entity.kind=="wallSpringLeft" else -1)


class RefillGoal(Goal):
    """Reach a consumed refill event, including regained dash in midair."""
    def achieved(self,state):
        key=self.id.removeprefix("refill:")
        return (not state.dead and state.room==self.room and state.dashes>0
                and any(identity==key and timer>0 for identity,timer in state.refill_timers))


class UnderhangGoal(Goal):
    """Pass below the obstruction, not merely near its upper side."""
    def achieved(self,state):
        return (not state.dead and state.room==self.room and abs(state.x-self.x)<14
                and self.y<=state.y<self.y+20)


def search_goal(simulator, start, goal: Goal, *, deadline: float,
                max_expansions=5000, max_frames=900, variant=0,
                allow_unsupported=False) -> SearchResult:
    """Search physical routes through reusable semantic support milestones.

    Some variants target an intermediate landing before continuing the original
    goal. Supports are read from geometry, never room-specific route scripts.
    Other variants retain direct search, so a poor decomposition cannot exclude
    a useful uninterrupted flight.
    """
    variant%=12
    data = simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    if goal.kind=="berry" and any(berry_id(goal.room,e)==goal.id for e in winged_berries(data.rooms[goal.room])):
        if goal.id in start.fled and goal.id not in start.berries:
            return SearchResult([],start,False,0,0,"winged_berry_fled_current_room_respawn_needed")
        goal=replace(goal,no_dash_until=goal.id)
    heuristic = Heuristic(data,goal)
    refills=[]
    if variant%3==2 and goal.kind in {"berry","room","target"} and max_expansions>=3:
        for entity in data.rooms[start.room].entities:
            identity=f"{start.room}:{entity.id}"
            if entity.kind!="refill" or identity in dict(start.refill_timers):
                continue
            distance=math.hypot(start.x-entity.x,start.y-entity.y)
            probe=replace(start,x=entity.x,y=entity.y+5)
            progress=heuristic(start)-heuristic(probe)
            if 24<distance<400 and progress>24:
                refills.append((distance-progress*.25,entity.id,entity))
    if refills and deadline>time.monotonic():
        entity=min(refills,key=lambda row:row[:2])[-1]
        refill=RefillGoal("refill",start.room,entity.x,entity.y+5,
                          f"refill:{start.room}:{entity.id}",goal.no_dash_until)
        now=time.monotonic()
        first=search_leg(simulator,start,refill,deadline=now+(deadline-now)*.55,
                         max_expansions=max(1,max_expansions*2//3),max_frames=max_frames,
                         variant=variant,allow_unsupported=allow_unsupported)
        if first.reached and len(first.actions)<max_frames:
            second=search_goal(simulator,first.state,goal,deadline=deadline,
                         max_expansions=max(1,max_expansions-first.expansions),
                         max_frames=max_frames-len(first.actions),variant=variant,
                         allow_unsupported=allow_unsupported)
            return SearchResult(first.actions+second.actions,second.state,second.reached,
                                first.expansions+second.expansions,
                                first.simulated_frames+second.simulated_frames,second.reason)
        fallback=_search_semantic_goal(simulator,start,goal,deadline=deadline,
                         max_expansions=max(1,max_expansions-first.expansions),
                         max_frames=max_frames,variant=variant,allow_unsupported=allow_unsupported)
        return replace(fallback,expansions=first.expansions+fallback.expansions,
                       simulated_frames=first.simulated_frames+fallback.simulated_frames)
    candidates = []
    if (variant%3 != 0 and goal.kind in {"berry","room","target"} and not start.following
            and max_expansions >= 3 and deadline > time.monotonic()):
        initial_h = heuristic(start)
        for surface in coverage_goals(data.rooms[start.room]):
            distance = math.hypot(start.x-surface.x,start.y-surface.y)
            probe = replace(start,x=surface.x,y=surface.y,on_ground=True)
            progress = initial_h-heuristic(probe)
            if 32 < distance < 230 and surface.y >= start.y-128 and progress > 24:
                # Prefer a nearer reset of dash/stamina over skipping every
                # useful platform in pursuit of the geometric target.
                candidates.append((distance-progress*.25,surface.key,surface))
        # Staggered one-way platforms require stepping off an edge and steering
        # back underneath before moving toward the distant goal. These transient
        # air milestones are followed immediately, never used as idle anchors.
        room=data.rooms[start.room]
        for platform in room.entities:
            if (platform.kind!="jumpThru" or abs(platform.y-start.y)>24
                    or not platform.x-16<=start.x<=platform.x+platform.width+16):
                continue
            blockers=[e for e in room.hazard_entities if e.kind=="spikesUp"
                      and platform.y+16<=e.y<=platform.y+112
                      and e.x<=platform.x+platform.width+16 and e.x+e.width>=platform.x-16]
            if blockers:
                y=max(e.y for e in blockers)+28
                passage=UnderhangGoal("passage",start.room,platform.x+platform.width/2,y,
                             f"under:{start.room}:{platform.id}",goal.no_dash_until)
                distance=math.hypot(start.x-passage.x,start.y-passage.y)
                progress=initial_h-heuristic(replace(start,x=passage.x,y=passage.y))
                if progress>0:
                    candidates.append((distance-progress*.25-24,passage.key,passage))
    if candidates:
        candidates.sort(key=lambda row:row[:2])
        surface = replace(candidates[(variant//3)%min(3,len(candidates))][-1],
                          no_dash_until=goal.no_dash_until)
        now=time.monotonic()
        first=search_leg(simulator,start,surface,deadline=now+(deadline-now)*.55,
                         max_expansions=max(1,max_expansions//2),max_frames=max_frames,
                         variant=variant,allow_unsupported=allow_unsupported)
        if first.reached and len(first.actions)<max_frames:
            second=search_goal(simulator,first.state,goal,deadline=deadline,
                        max_expansions=max(1,max_expansions-first.expansions),
                        max_frames=max_frames-len(first.actions),variant=variant+1,
                        allow_unsupported=allow_unsupported)
            return SearchResult(first.actions+second.actions,second.state,second.reached,
                                first.expansions+second.expansions,
                                first.simulated_frames+second.simulated_frames,second.reason)
        fallback=_search_semantic_goal(simulator,start,goal,deadline=deadline,
                    max_expansions=max(1,max_expansions-first.expansions),max_frames=max_frames,
                    variant=variant,allow_unsupported=allow_unsupported)
        return replace(fallback,expansions=first.expansions+fallback.expansions,
                       simulated_frames=first.simulated_frames+fallback.simulated_frames)
    return _search_semantic_goal(simulator,start,goal,deadline=deadline,
                    max_expansions=max_expansions,max_frames=max_frames,variant=variant,
                    allow_unsupported=allow_unsupported)


def _search_semantic_goal(simulator, start, goal: Goal, *, deadline: float,
                max_expansions=5000, max_frames=900, variant=0,
                allow_unsupported=False) -> SearchResult:
    """Search a goal, optionally through an appropriate nearby spring.

    Parameters/return type match ``search_leg``. Expansion and frame limits
    apply across the component legs. If a spring attempt fails, direct search
    can use the remaining budget. A failed strategy never reports collection.
    """
    map_data = simulator.map_data if hasattr(simulator, "map_data") else simulator.map
    options = dict(max_expansions=max_expansions, max_frames=max_frames,
                   variant=variant, allow_unsupported=allow_unsupported)
    # Room entry is another semantic milestone. Searching all movement toward
    # a distant berry at once can bury a useful arrival state in fine motion
    # variants before room-local spring/collection strategies get a turn.
    if (goal.kind == "berry" and start.room != goal.room
            and goal.id not in start.following and not goal.achieved(start)
            and max_expansions >= 2 and deadline > time.monotonic()):
        now = time.monotonic()
        arrival = search_leg(
            simulator, start, Goal("room", goal.room, goal.x, goal.y,no_dash_until=goal.no_dash_until),
            deadline=min(deadline, now + max(0.05, (deadline-now)*0.35)),
            max_expansions=max(1, max_expansions//3), max_frames=max_frames,
            variant=variant, allow_unsupported=allow_unsupported,
        )
        remaining = max(0, max_expansions-arrival.expansions)
        if goal.achieved(arrival.state):
            return replace(arrival, reached=True, reason="goal")
        if arrival.reached and len(arrival.actions) < max_frames and remaining:
            collection = search_goal(
                simulator, arrival.state, goal, deadline=deadline,
                max_expansions=remaining, max_frames=max_frames-len(arrival.actions),
                variant=variant, allow_unsupported=allow_unsupported,
            )
            return SearchResult(
                arrival.actions+collection.actions, collection.state, collection.reached,
                arrival.expansions+collection.expansions,
                arrival.simulated_frames+collection.simulated_frames, collection.reason,
            )
        if remaining and time.monotonic() < deadline:
            fallback = search_leg(simulator, start, goal, deadline=deadline,
                                  max_expansions=remaining, max_frames=max_frames,
                                  variant=variant, allow_unsupported=allow_unsupported)
            return replace(fallback, expansions=arrival.expansions+fallback.expansions,
                           simulated_frames=arrival.simulated_frames+fallback.simulated_frames)
        return replace(arrival, reached=False,
                       reason="time_budget" if time.monotonic() >= deadline else "budget")

    candidates = []
    if (goal.kind == "berry" and start.room == goal.room
            and goal.id not in start.following and not goal.achieved(start)):
        for entity in map_data.rooms[start.room].entities:
            if entity.kind != "spring" or not usable_spring(entity):
                continue
            approach = math.hypot(start.x - entity.x, start.y - entity.y)
            if entity.y - goal.y > 72 and approach <= 240:
                candidates.append((approach, abs(entity.x-goal.x), entity.id, entity))
    if not candidates or deadline <= time.monotonic() or max_expansions < 2:
        return search_leg(simulator, start, goal, deadline=deadline, **options)

    spring = min(candidates, key=lambda item: item[:3])[-1]
    spring_goal = SpringGoal("spring", start.room, spring.x, spring.y,
                             f"spring:{start.room}:{spring.id}",goal.no_dash_until)
    now = time.monotonic()
    # Leave most of the budget for the actual berry; a failed spring approach
    # must not consume the entire request before direct search can be tried.
    spring_deadline = min(deadline, now + max(0.05, (deadline-now) * 0.25))
    first = search_leg(simulator, start, spring_goal, deadline=spring_deadline,
                       max_expansions=max(1, max_expansions//3),
                       max_frames=max_frames, variant=variant,
                       allow_unsupported=allow_unsupported)
    total_expansions, total_simulated = first.expansions, first.simulated_frames
    remaining_expansions = max(0, max_expansions-total_expansions)
    if goal.achieved(first.state):
        return replace(first, reached=True, reason="goal")
    if first.reached and len(first.actions) < max_frames and remaining_expansions:
        second = search_leg(simulator, first.state, goal, deadline=deadline,
                            max_expansions=remaining_expansions,
                            max_frames=max_frames-len(first.actions), variant=variant,
                            allow_unsupported=allow_unsupported)
        return SearchResult(
            first.actions + second.actions, second.state, second.reached,
            total_expansions + second.expansions,
            total_simulated + second.simulated_frames, second.reason,
        )

    if remaining_expansions and time.monotonic() < deadline:
        fallback = search_leg(simulator, start, goal, deadline=deadline,
                              max_expansions=remaining_expansions,
                              max_frames=max_frames, variant=variant,
                              allow_unsupported=allow_unsupported)
        return replace(fallback, expansions=total_expansions+fallback.expansions,
                       simulated_frames=total_simulated+fallback.simulated_frames)

    # A spring approach is only a partial candidate for the berry goal.
    # The first leg may incidentally collect it, so check the requested goal.
    reached = goal.achieved(first.state)
    return replace(first, reached=reached, reason="goal" if reached else "time_budget" if time.monotonic() >= deadline else "budget")
