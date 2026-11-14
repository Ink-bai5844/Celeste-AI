"""Observed exploration ledger; static inventory is never counted as progress."""
from __future__ import annotations

from collections import Counter, deque
import math


class Exploration:
    def __init__(self, saved=None):
        saved=saved or {}
        self.cells={tuple(cell) for cell in saved.get("visited_cells",[])}
        self.room_visits=Counter(saved.get("room_visits",{}))
        self.edge_visits=Counter(saved.get("edge_visits",{}))
        self.failed_goals=Counter(saved.get("failed_goals",{}))
        self.goal_deferrals=Counter(saved.get("goal_deferrals",{}))
        self.failed_regions=deque(saved.get("failed_regions",[]),maxlen=64)
        self.seen_breakable_walls=set(saved.get("seen_breakable_walls",[]))
        self.opened_walls=set(saved.get("opened_walls",[]))
        self.active_destination=saved.get("active_destination")
        self.active_berry=saved.get("active_berry")
        self.destination_deferred=Counter(saved.get("destination_deferred",{}))
        self.recent_edges=deque(saved.get("recent_edges",[]),maxlen=16)
        self.settled_rooms=set(saved.get("settled_rooms",[]))
        self.navigation_revision=int(saved.get("navigation_revision",0))
        self.stalled_edges=Counter(saved.get("stalled_edges",{}))
        self.arrivals={}
        self.last_room=None
        self.last_spawn=None
        self.last_progress_frame=int(saved.get("last_progress_frame",0))

    def observe(self,obs):
        if self.active_berry in obs.get("berries",[]):
            self.active_berry=None
            self.navigation_revision+=1
        room=obs["room"]
        spawn=obs.get("spawn_id")
        if room!=self.last_room:
            self.room_visits[room]+=1
            if self.last_room is not None and spawn==self.last_spawn:
                edge=f"{self.last_room}>{room}"
                self.edge_visits[edge]+=1
                self.recent_edges.append(edge)
                signature=(room,round(obs["x"]/4),round(obs["y"]/4),obs.get("dashes"),
                           round(obs.get("vx",0)/30),round(obs.get("vy",0)/30),self.active_destination)
                progress=(len(self.cells),tuple(sorted(obs.get("berries",[]))),len(self.opened_walls))
                if self.arrivals.get(signature)==progress:
                    self.stalled_edges[edge]+=1
                self.arrivals[signature]=progress
            self.last_room=room
        self.last_spawn=spawn
        cell=(room,math.floor(obs["x"]/16),math.floor(obs["y"]/16))
        if cell not in self.cells:
            self.cells.add(cell)
            self.last_progress_frame=obs["frame"]
        environment=obs.get("room_entities",{})
        if isinstance(environment,dict):
            absent=set(environment.get("do_not_load",[]))
            for entity in environment.get("entities",[]):
                if entity.get("kind")=="dashBlock" and entity.get("id"):
                    identity=entity["id"]
                    self.seen_breakable_walls.add(identity)
                    if entity.get("present") is False or identity in absent:
                        self.opened_walls.add(identity)
            self.opened_walls.update(self.seen_breakable_walls & absent)

    def set_destination(self,room):
        if room and room!=self.active_destination:
            self.active_destination=room
            self.navigation_revision+=1

    def set_berry(self,identity):
        if identity and identity!=self.active_berry:
            self.active_berry=identity
            self.navigation_revision+=1

    def defer_goal(self,goal):
        self.goal_deferrals[goal]+=1
        if self.active_berry==goal:
            self.active_berry=None
        if self.active_destination:
            self.destination_deferred[self.active_destination]+=1
        self.active_destination=None
        self.navigation_revision+=1

    def mark_settled(self,room):
        self.settled_rooms.add(room)
        if self.active_destination==room:
            self.active_destination=None
            self.navigation_revision+=1

    def reject(self,obs,reason,goal=None):
        if isinstance(goal,dict):
            goal=goal.get("goal") or goal.get("key")
        if goal:
            self.failed_goals[str(goal)]+=1
        self.failed_regions.append({"room":obs["room"],"x":obs["x"],"y":obs["y"],"reason":reason})

    def snapshot(self):
        return {"visited_cells":sorted(self.cells),"room_visits":dict(self.room_visits),
                "edge_visits":dict(self.edge_visits),"failed_goals":dict(self.failed_goals),
                "goal_deferrals":dict(self.goal_deferrals),
                "failed_regions":list(self.failed_regions),
                "seen_breakable_walls":sorted(self.seen_breakable_walls),
                "opened_walls":sorted(self.opened_walls),
                "active_destination":self.active_destination,
                "active_berry":self.active_berry,
                "destination_deferred":dict(self.destination_deferred),
                "recent_edges":list(self.recent_edges),"settled_rooms":sorted(self.settled_rooms),
                "navigation_revision":self.navigation_revision,"stalled_edges":dict(self.stalled_edges),
                "last_progress_frame":self.last_progress_frame}
