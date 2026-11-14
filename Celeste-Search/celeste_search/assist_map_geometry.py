"""Observed minimap geometry, independent of planning and player control."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .runtime_state import (cassette_snapshot_from_observation,
                            falling_snapshot_from_observation, RuntimeStateError)
from .physics import _spike_rect, _spike_rides, _truth

MOVING_MAP_KINDS = frozenset({"zipMover", "fallingBlock", "crumbleBlock", "dashBlock", "cassetteBlock"})

def map_hazard_rect(entity):
    """Use the same directional spike bounds as collision simulation."""
    if entity.kind in {"spikesUp","spikesDown","spikesLeft","spikesRight"}:
        left,top,right,bottom=_spike_rect(entity)
        return left,top,right-left,bottom-top
    return entity.x,entity.y,max(4,entity.width),max(3,entity.height)


def moving_spike_owners(room):
    """Match the simulator's first native-style StaticMover attachment."""
    owners={}
    movers=[e for e in room.entities if e.kind in {"zipMover","fallingBlock","dashBlock"}]
    for hazard in room.entities:
        if not hazard.kind.startswith("spikes"):
            continue
        owner=next((mover for mover in movers if _spike_rides(hazard,mover)),None)
        if owner is not None:
            owners[(hazard.kind,hazard.id)]=owner
    return owners


def falling_spike_owners(room):
    return {key: owner for key, owner in moving_spike_owners(room).items()
            if owner.kind == "fallingBlock"}


@dataclass(frozen=True)
class FallingMapGeometry:
    identity: str
    rect: tuple[float,float,float,float] | None
    hazards: tuple[tuple[float,float,float,float],...]
    known: bool
    phase: int | None


def falling_map_geometry(map_data,room_name,observation):
    """Render one complete observation, never a prediction or a prior room visit.

    An unknown block is an outlined map reference. Explicit native removal is
    different: both its body and its attached hazards disappear. Decode Single
    values plus remainders first, then project to the same integer collision
    bounds as the simulator; the pending subpixel motion is not an actual edge.
    """
    room=map_data.rooms[room_name]
    records={}
    if isinstance(observation,dict) and observation.get("room")==room_name:
        try:
            records={row[0]:row for row in falling_snapshot_from_observation(map_data,observation)}
        except (RuntimeStateError,ValueError,TypeError,KeyError):
            pass
    owners=falling_spike_owners(room)
    items=[]
    for entity in room.entities:
        if entity.kind!="fallingBlock":
            continue
        identity=f"{room.name}:{entity.id}"
        record=records.get(identity)
        phase=record[1] if record is not None else None
        if phase==4:
            items.append(FallingMapGeometry(identity,None,(),True,phase))
            continue
        x,y=(round(record[3]),round(record[4])) if record is not None else (entity.x,entity.y)
        hazards=[]
        for hazard in room.entities:
            if owners.get((hazard.kind,hazard.id))==entity:
                hx,hy,w,h=map_hazard_rect(hazard)
                hazards.append((hx+x-entity.x,hy+y-entity.y,w,h))
        items.append(FallingMapGeometry(identity,(x,y,entity.width,entity.height),
                                       tuple(hazards),record is not None,phase))
    return tuple(items)


@dataclass(frozen=True)
class MovingMapGeometry:
    identity: str
    kind: str
    rect: tuple[float,float,float,float] | None
    hazards: tuple[tuple[float,float,float,float],...]
    known: bool
    solid: bool
    color: str
    dash_breakable: bool = False


def _finite(value):
    try:
        return type(value) in (float,int) and math.isfinite(value)
    except (OverflowError,ValueError):
        return False


def _geometry_record(record, entity, room):
    """Validate only observed geometry; an unknown coroutine is not a prediction."""
    if (not isinstance(record,dict) or record.get("id") != f"{room.name}:{entity.id}"
            or record.get("room") != room.name or record.get("kind") != entity.kind
            or type(record.get("present")) is not bool):
        return None
    # Present=false is current scene membership, even if the last coordinates
    # were retained after collection/destruction. Never display those coordinates.
    if record["present"] is False:
        return (None,False)
    if (record.get("retained_observation") is not False
            or type(record.get("active")) is not bool
            or type(record.get("collidable")) is not bool):
        return None
    keys=("left","top","width","height")
    if any(not _finite(record.get(key)) for key in (*keys,"x","y")):
        return None
    if record["width"] <= 0 or record["height"] <= 0:
        return None
    if not record["collidable"] and entity.kind in {"dashBlock","crumbleBlock"}:
        return (None,False)
    # Collider Left/Top already contain actual native motion. Solid remainder
    # is deferred movement and must not be added to the displayed rectangle.
    return tuple(record[key] for key in keys),record["collidable"]


def moving_map_geometry(map_data,room_name,observation):
    """One fresh room snapshot replaces every movable body and attached hazard.

    Missing/duplicate records make that kind unknown, rather than mixing a
    partial snapshot with old positions. Unknowns are map references; explicit
    disappearance clears the body and its owned hazards. No state is cached.
    """
    room=map_data.rooms[room_name]
    raw={}
    snapshot=observation.get("room_entities") if isinstance(observation,dict) else None
    if (isinstance(observation,dict) and observation.get("room")==room_name
            and isinstance(snapshot,dict) and isinstance(snapshot.get("entities"),list)
            and snapshot.get("room",room_name)==room_name
            and ("frame" not in snapshot or snapshot["frame"]==observation.get("frame"))):
        for record in snapshot["entities"]:
            if isinstance(record,dict) and isinstance(record.get("id"),str):
                raw.setdefault(record["id"],[]).append(record)
    complete={kind:all(len(raw.get(f"{room_name}:{e.id}",[]))==1
                       for e in room.entities if e.kind==kind) for kind in MOVING_MAP_KINDS}
    falling={item.identity:item for item in falling_map_geometry(map_data,room_name,
             observation if raw else None)}
    stopped=False
    if any(e.kind=="cassetteBlock" for e in room.entities) and raw:
        try:
            stopped=cassette_snapshot_from_observation(map_data,observation).get("cassette_stopped",False)
        except (RuntimeStateError,ValueError,TypeError,KeyError):
            pass
    owners=moving_spike_owners(room)
    colors=("#49aaf0","#f049be","#f3d85e","#83d85d")
    items=[]
    for entity in room.entities:
        if entity.kind not in MOVING_MAP_KINDS:
            continue
        identity=f"{room_name}:{entity.id}"
        dash_breakable=entity.kind=="dashBlock" and _truth(entity.attrs.get("canDash",True))
        color="#d4a65c" if dash_breakable else "#647c91"
        if entity.kind=="cassetteBlock":
            color=colors[int(entity.attrs.get("index",0))%len(colors)]
        if entity.kind=="fallingBlock":
            item=falling[identity]
            items.append(MovingMapGeometry(identity,entity.kind,item.rect,item.hazards,
                                           item.known,item.known and item.rect is not None,color))
            continue
        record=raw.get(identity,[None])[0] if complete[entity.kind] else None
        observed=_geometry_record(record,entity,room)
        if entity.kind=="cassetteBlock" and stopped:
            observed=(None,False)
        known=observed is not None
        rect,solid=observed if known else ((entity.x,entity.y,entity.width,entity.height),False)
        if known and entity.kind=="dashBlock" and record.get("can_dash") is False:
            dash_breakable=False
            color="#647c91"
        hazards=[]
        if rect is not None:
            # Offset attachment from the entity anchor, not from a collider's
            # possible local offset. Fresh anchors are checked independently.
            anchor=(record["x"],record["y"]) if known else (entity.x,entity.y)
            for hazard in room.entities:
                if owners.get((hazard.kind,hazard.id))==entity:
                    hx,hy,w,h=map_hazard_rect(hazard)
                    hazards.append((hx+anchor[0]-entity.x,hy+anchor[1]-entity.y,w,h))
        items.append(MovingMapGeometry(identity,entity.kind,rect,tuple(hazards),known,solid,color,
                                       dash_breakable))
    return tuple(items)

