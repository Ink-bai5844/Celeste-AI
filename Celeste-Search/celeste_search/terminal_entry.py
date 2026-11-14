"""Certify only a final directed crossing into a locally clear entry corridor.

Unmodeled cassette and lookout interactions remain unsupported. The certificate covers one boundary
event and never permits another simulated input in the destination room. Caller
must attach it only to the final public route point and stop on that exact hit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math

from .assist_targets import expand_exit_target, room_exit_targets, transition_target_reached


@dataclass(frozen=True)
class TerminalEntryCertificate:
    map_token: int
    source_room: str
    destination_room: str
    edge: str
    tile_x: float
    tile_y: float
    arrival_bounds: tuple[float, float, float, float]
    clearance_bounds: tuple[float, float, float, float]
    unsupported_kinds: tuple[str, ...]
    lanes: tuple[TerminalEntryCertificate, ...] = ()

    def target(self):
        target = dict(room=self.source_room, x=self.tile_x, y=self.tile_y,
                      tile_size=8, edge=self.edge, transition_to=self.destination_room,
                      arrival_bounds=self.arrival_bounds)
        if self.lanes:
            target["exit_region"] = [dict(x=lane.tile_x, y=lane.tile_y,
                                          arrival_bounds=lane.arrival_bounds)
                                     for lane in self.lanes]
        return target


def _data(simulator):
    return simulator.map_data if hasattr(simulator, "map_data") else simulator.map


def _unsupported(report):
    return set(report.get("unsupported", report.get("unsupported_entities", ())))


def _overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _local_unknown_bounds(entity):
    values = (entity.x, entity.y, entity.width, entity.height)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        return None
    if entity.kind == "cassetteBlock":
        # Vanilla pre-toggle/activation moves the body vertically by a pixel;
        # inactive blocks also sit lower. Cover all those phases and probes.
        if entity.width <= 0 or entity.height <= 0 or entity.nodes:
            return None
        return (entity.x, entity.y - 4, entity.x + entity.width, entity.y + entity.height + 4)
    if entity.kind == "cassette":
        # Conservative pickup/bob envelope. Its nodes describe the subsequent
        # collection flight; no collection can trigger outside this envelope.
        return (entity.x - 24, entity.y - 24, entity.x + 24, entity.y + 24)
    if entity.kind == "towerviewer":
        # Vanilla Lookout is not a Solid, but its TalkComponent can consume
        # Dash (the default shared binding) and enter player state 11 without
        # Level.InCutscene. Enclose both its 4x4 collider and 48x8 talk area
        # with margin; permit only an entry and passive tail outside it.
        return (entity.x - 28, entity.y - 16, entity.x + 28, entity.y + 8)
    return None


def terminal_entry_certificate(simulator, target, *, is_terminal: bool):
    """Return narrow permission for a canonical last exit, otherwise None."""
    if is_terminal is not True or not isinstance(target, dict):
        return None
    data = _data(simulator)
    source, destination = target.get("room"), target.get("transition_to")
    if (not isinstance(source, str) or not isinstance(destination, str)
            or source not in data.rooms or destination not in data.rooms or source == destination):
        return None
    supplied_bounds = target.get("arrival_bounds")
    if (not isinstance(supplied_bounds, (tuple, list)) or len(supplied_bounds) != 4
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in supplied_bounds)):
        return None
    source_report = simulator.capability_report(source)
    if _unsupported(source_report) or source_report.get("unsupported_collectibles"):
        return None
    report = simulator.capability_report(destination)
    unknown = _unsupported(report)
    if (not unknown or not unknown <= {"cassette", "cassetteBlock", "towerviewer"}
            or report.get("unsupported_collectibles")
            or any(name.endswith(":runtime_script") for name in report.get("approximate_entities", {}))):
        return None
    # Canonical metadata prevents a caller from expanding a lane or declaring
    # an interior tile to be an exit. This also checks permanent portal geometry.
    keys = ("room", "x", "y", "tile_size", "edge", "transition_to")
    canonical = next((item for item in room_exit_targets(data, source)
                      if all(target.get(key) == item[key] for key in keys)
                      and tuple(supplied_bounds) == tuple(item["arrival_bounds"])), None)
    if canonical is None:
        return None
    members = [canonical]
    if "exit_region" in target:
        members = expand_exit_target(data, canonical)["exit_region"]
        supplied = target["exit_region"]
        if (not isinstance(supplied, (tuple, list)) or len(supplied) != len(members)
                or any(not isinstance(actual, dict)
                       or actual.get("x") != expected["x"] or actual.get("y") != expected["y"]
                       or not isinstance(actual.get("arrival_bounds"), (tuple, list))
                       or tuple(actual["arrival_bounds"]) != tuple(expected["arrival_bounds"])
                       for actual, expected in zip(supplied, members))):
            return None
    rectangles = [_local_unknown_bounds(entity) for entity in data.rooms[destination].entities
                  if entity.kind in unknown]
    if any(rectangle is None for rectangle in rectangles):
        return None
    certificates = []
    for member in members:
        bounds = tuple(float(value) for value in member["arrival_bounds"])
        # Check each actual lane independently. A safe opening must never lend
        # permission to another lane overlapping an unknown entity's envelope.
        clearance = (bounds[0] - 5, bounds[1] - 6.5, bounds[2] + 5, bounds[3] + 6.5)
        if any(_overlap(clearance, rectangle) for rectangle in rectangles):
            continue
        certificates.append(TerminalEntryCertificate(
            id(data), source, destination, canonical["edge"],
            float(member["x"]), float(member["y"]), bounds, clearance, tuple(sorted(unknown))))
    if not certificates:
        return None
    return (replace(certificates[0], lanes=tuple(certificates))
            if "exit_region" in target else certificates[0])


def terminal_entry_crossing_allowed(simulator, certificate, before, after):
    """Admit this single simulated crossing, never a destination-room state."""
    if not isinstance(certificate, TerminalEntryCertificate):
        return False
    if certificate.lanes:
        return any(terminal_entry_crossing_allowed(simulator, lane, before, after)
                   for lane in certificate.lanes)
    if (certificate.map_token != id(_data(simulator))
            or before.dead or after.dead or before.room != certificate.source_room
            or after.room != certificate.destination_room or after.reason != "transition"
            or not transition_target_reached(after, certificate.target())):
        return False
    # The certificate excludes unknown bounds; supported geometry/hazards are
    # still checked at the actual native-positioned arrival, including movers.
    if simulator.solid_at(after.room, after.x, after.y, ducking=after.ducking, state=after):
        return False
    return not simulator._hazard_hit(after) and terminal_entry_handoff_safe(simulator, certificate, after)


@lru_cache(maxsize=2048)
def terminal_entry_handoff_safe(simulator, certificate, arrival):
    """Prove passive landing using only geometry independent of unknown beats.

    These neutral frames are an offline release check, never appended to the
    executable route. Unknown blocks are excluded for the entire swept body,
    under every beat phase; reaching their vicinity rejects the certificate.
    A separate simulator keeps this bounded check out of the search frontier.
    """
    from .physics import Simulator
    data = _data(simulator)
    if not isinstance(certificate, TerminalEntryCertificate):
        return False
    if certificate.lanes:
        return any(transition_target_reached(arrival, lane.target())
                   and terminal_entry_handoff_safe(simulator, lane, arrival)
                   for lane in certificate.lanes)
    if (certificate.map_token != id(data) or arrival.dead
            or arrival.room != certificate.destination_room):
        return False
    unknown = []
    for entity in data.rooms[arrival.room].entities:
        if entity.kind in certificate.unsupported_kinds:
            bounds = _local_unknown_bounds(entity)
            if bounds is None:
                return False
            unknown.append(bounds)
    passive = Simulator(data, max_dashes=simulator.max_dashes)
    state = arrival
    stable = 0
    for _ in range(120):
        # A player's largest body plus corner/contact probes. Conservative
        # bounds include attached spikes and cassette pickup/bob envelopes.
        body = (state.x - 8, state.y - 15, state.x + 8, state.y + 4)
        if any(_overlap(body, bounds) for bounds in unknown):
            return False
        following = passive.step(state, 0)
        if following.dead or following.room != arrival.room:
            return False
        swept = (min(state.x, following.x) - 8, min(state.y, following.y) - 15,
                 max(state.x, following.x) + 8, max(state.y, following.y) + 4)
        if any(_overlap(swept, bounds) for bounds in unknown):
            return False
        permanent_support = (passive.solid_at(following.room, following.x, following.y + 1,
                                             ducking=following.ducking, dynamic_rects=())
                             or passive._jumpthru_at(following.room, following.x,
                                                    following.y, following.y + 1))
        stable = (stable + 1 if following.on_ground and permanent_support and following.state == 0
                  and following.x == state.x and following.y == state.y
                  and abs(following.vx) < .01 and abs(following.vy) < .01 else 0)
        state = following
    return stable >= 12
