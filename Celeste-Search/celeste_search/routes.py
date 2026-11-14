"""Portable route artifacts; predicted and observed achievements stay separate."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

SCHEMA = "celeste-offline-route-v1"
SOURCES = [
    "https://github.com/NoelFB/Celeste/blob/master/Source/Player/Player.cs",
    "https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/BinaryPacker.cs",
    "https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/Strawberry.cs",
]


def map_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def encode_actions(actions: Iterable[int]) -> list[dict]:
    runs: list[dict] = []
    for mask in actions:
        if not isinstance(mask, int) or not 0 <= mask < 128:
            raise ValueError(f"Invalid action mask: {mask!r}")
        if runs and runs[-1]["mask"] == mask:
            runs[-1]["frames"] += 1
        else:
            runs.append({"mask": mask, "frames": 1})
    return runs


def decode_actions(runs: list[dict], limit: int = 2_000_000) -> list[int]:
    result = []
    for run in runs:
        mask, frames = run["mask"], run["frames"]
        if type(mask) is not int or not 0 <= mask < 128:
            raise ValueError("Route mask must be an integer in 0..127")
        if type(frames) is not int or frames <= 0 or len(result) + frames > limit:
            raise ValueError("Invalid or excessively long route")
        result.extend([mask] * frames)
    return result


def snapshot(state) -> dict:
    result = {"frame": state.frame, "room": state.room,
            "x": round(state.x, 4), "y": round(state.y, 4),
            "vx": round(state.vx, 4), "vy": round(state.vy, 4),
            "dashes": state.dashes, "on_ground": state.on_ground,
            "dead": state.dead, "berries": sorted(state.berries),
            "following": sorted(state.following)}
    if state.mechanisms:
        # Keep exact mechanism positions/timers. Execution selects only the
        # entity kinds with a verified native decoder, using the loaded map.
        result["mechanisms"]=[list(record) for record in state.mechanisms]
    if getattr(state,"cassette_clock",None) is not None or getattr(state,"cassette_stopped",False):
        clock = state.cassette_clock.to_dict() if state.cassette_clock is not None else None
        if clock is not None:
            clock.pop("update_order",None)
        blocks = []
        for block in state.cassette_blocks:
            value = block.to_dict()
            value.pop("update_order",None)
            blocks.append(value)
        result.update(cassette_clock=clock,cassette_blocks=blocks,
                      player_depth=state.player_depth,player_actual_depth=state.player_actual_depth)
        if getattr(state,"cassette_stopped",False):
            result["cassette_stopped"] = True
    return result


def json_state(state) -> dict:
    return {key: sorted(value) if isinstance(value, (set, frozenset)) else value
            for key, value in asdict(state).items()}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_route(path: Path) -> dict:
    route = json.loads(path.read_text(encoding="utf-8"))
    if route.get("schema") != SCHEMA:
        raise ValueError("Unsupported route schema")
    actions = decode_actions(route["actions"])
    if len(actions) != route["frames"]:
        raise ValueError("Route frame count disagrees with actions")
    return route


def trace_route(simulator, start_room: str, actions: list[int], interval: int = 10):
    state = simulator.initial_state(start_room)
    samples, rooms, cells = [snapshot(state)], {state.room}, set()
    for index, mask in enumerate(actions, 1):
        old_room, old_berries = state.room, state.berries
        state = simulator.step(state, mask)
        if state.dead:
            raise ValueError(f"Offline route dies at frame {index}")
        rooms.add(state.room)
        cells.add((state.room, int(state.x // 16), int(state.y // 16)))
        if index % interval == 0 or state.room != old_room or state.berries != old_berries:
            samples.append(snapshot(state))
    if samples[-1]["frame"] != state.frame:
        samples.append(snapshot(state))
    return state, samples, rooms, cells
