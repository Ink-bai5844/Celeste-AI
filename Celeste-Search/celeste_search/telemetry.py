"""Asynchronous append-only execution evidence, outside the game input loop."""
from __future__ import annotations

import json
from pathlib import Path
from queue import Queue, Full
from threading import Thread


class Telemetry:
    def __init__(self, directory:Path):
        self.directory=directory
        directory.mkdir(parents=True,exist_ok=True)
        self.queue=Queue(maxsize=8192)
        self.dropped=0
        self.error=None
        self.thread=Thread(target=self._write,daemon=True,name="celeste-log-writer")
        self.thread.start()

    def emit(self,kind,data):
        try:
            self.queue.put_nowait((kind,data))
        except Full:
            self.dropped+=1

    def _write(self):
        try:
            with (self.directory/"frames.jsonl").open("a",encoding="utf-8",buffering=262144) as frames, (self.directory/"events.jsonl").open("a",encoding="utf-8",buffering=65536) as events:
                while True:
                    item=self.queue.get()
                    if item is None:
                        break
                    kind,data=item
                    if kind=="plan":
                        path=self.directory/(f"plan-{data['run_plan_id']:05d}.json")
                        path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
                    else:
                        target=frames if kind=="frame" else events
                        target.write(json.dumps(data,ensure_ascii=False,separators=(",",":"))+"\n")
                        if kind!="frame":
                            target.flush()
        except Exception as error:
            self.error=f"{type(error).__name__}: {error}"

    def close(self):
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join(timeout=10)


def compact_frame(obs,mask,plan_id,index,expected=None):
    return {"frame":obs["frame"],"room":obs["room"],"spawn_id":obs.get("spawn_id"),
            "x":obs["x"],"y":obs["y"],"vx":obs["vx"],"vy":obs["vy"],
            "state":obs["state"],"dashes":obs["dashes"],"on_ground":obs["on_ground"],
            "berries":obs["berries"],"following":obs["following"],"mask":mask,
            "plan_id":plan_id,"plan_index":index,"expected":expected,
            "engine_frame":obs.get("engine_frame"),"dt":obs.get("delta_time"),
            "bridge_roundtrip_ms":obs.get("previous_round_trip_ms"),
            "model_state":obs.get("model_state",{})}
