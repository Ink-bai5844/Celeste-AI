from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

from .maps import load_map, discover_maps
from .routes import SCHEMA, SOURCES, encode_actions, decode_actions, load_route, map_hash, snapshot, trace_route, write_json
from .game_platform import default_game_dir, normalize_game_dir

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPS = default_game_dir() / "Content" / "Maps"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs"


def _worker(config):
    # Worker imports cannot start the game or access the socket playback module.
    from .physics import Simulator
    from .planner import Goal, Heuristic, plan_campaign
    map_data = load_map(Path(config["map"]))
    simulator = Simulator(map_data)
    target = Goal(**config["target"]) if config.get("target") else None
    result = plan_campaign(simulator,map_data,config["room"],seconds=config["seconds"],
                           expansions=config["expansions"],max_goals=config["max_goals"],
                           variant=config["variant"],allow_unsupported=config["allow_unsupported"],
                           target=target,prefix=config.get("prefix"))
    actions = result.pop("actions")
    result.pop("state")
    state,samples,rooms,cells = trace_route(simulator,config["room"],actions)
    return {"schema":SCHEMA,"simulator_version":"0.1.0",
            "simulator_sha256":map_hash(Path(__file__).with_name("physics.py")),
            "route_scope":"chapter_from_start" if config["room"]==map_data.start_room().name else "room_practice",
            "map":{"path":str(map_data.source.resolve()),"sha256":map_hash(map_data.source),"package":map_data.package},
            "start_room":config["room"],"frames":len(actions),"actions":encode_actions(actions),
            "predicted":{"berries":sorted(state.berries),"rooms":sorted(rooms),
                         "coverage_cells":sorted(cells),"state":snapshot(state)},
            "trace":samples,"search":result,"capabilities":simulator.capability_report(),
            "target":asdict(target) if target else None,
            "target_reached":target.achieved(state) if target else None,
            "target_remaining_heuristic":Heuristic(map_data,target)(state) if target else None,
            "allow_unsupported":config["allow_unsupported"],"runtime_verified":False,
            "sources":SOURCES}


def _score(route):
    predicted = route["predicted"]
    target_done = sum(leg["reached"] for leg in route["search"]["legs"])
    if route.get("target"):
        return (route["target_reached"],-route["target_remaining_heuristic"],-route["frames"])
    return (len(predicted["berries"]),len(predicted["rooms"]),target_done,
            len(predicted["coverage_cells"]),-route["frames"])


def _target(value, data, room):
    from .planner import Goal
    if not value:
        return None
    parts = value.split(":")
    if parts[0]=="room" and len(parts)==2:
        dest = data.rooms[parts[1]]
        if not dest.spawns:
            raise ValueError(f"Room {dest.name!r} has no playable spawn")
        return Goal("room",dest.name,*dest.spawns[0])
    if parts[0]=="berry" and len(parts)==3:
        dest = data.rooms[parts[1]]
        entity = next((e for e in dest.berries if str(e.id)==parts[2]),None)
        if entity is None:
            raise ValueError(f"Unknown berry: {value}")
        if entity.berry_category not in ("ordinary","winged"):
            raise ValueError(f"Berry category {entity.berry_category!r} is not implemented")
        return Goal("berry",dest.name,entity.x,entity.y,f"{dest.name}:{entity.id}")
    if parts[0]=="point" and len(parts)==3:
        x,y=float(parts[1]),float(parts[2])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Target coordinates must be finite")
        return Goal("explore",room,x,y)
    raise ValueError("Target format: room:NAME, berry:ROOM:ID, or point:WORLD_X:WORLD_Y")


def plan(args):
    data = load_map(args.map.resolve())
    room = args.room or data.start_room().name
    if room not in data.rooms or not data.rooms[room].spawns:
        raise ValueError(f"Room {room!r} has no player spawn")
    target = _target(args.target,data,room)
    prefix = []
    if args.resume:
        prior = load_route(args.resume)
        if prior["map"]["sha256"]!=map_hash(data.source) or prior["start_room"]!=room:
            raise ValueError("Resume map hash/start room does not match")
        if prior.get("simulator_sha256")!=map_hash(Path(__file__).with_name("physics.py")):
            raise ValueError("Simulator changed since this route was generated; recompute the route")
        prefix = decode_actions(prior["actions"])
    config = {"map":str(data.source),"room":room,"seconds":args.seconds,
              "expansions":args.expansions,"max_goals":args.max_goals,
              "target":asdict(target) if target else None,"prefix":prefix,
              "allow_unsupported":args.allow_unsupported}
    print(f"Offline search: {data.package}, room {room}, {args.workers} worker(s), {args.seconds:g}s each",flush=True)
    variants=[]
    if args.workers==1:
        variants.append(_worker(dict(config,variant=0)))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(_worker,dict(config,variant=i)) for i in range(args.workers)]
            for future in as_completed(futures):
                route=future.result()
                variants.append(route)
                print(f"Candidate {route['search']['variant']}: {route['frames']} frames; "
                      f"predicted berries={len(route['predicted']['berries'])}, rooms={len(route['predicted']['rooms'])}",flush=True)
    best=max(variants,key=_score)
    output=args.output.resolve()
    write_json(output,best)
    for route in variants:
        if route is not best:
            write_json(output.with_name(output.stem+f".candidate-{route['search']['variant']}.json"),route)
    from .report import write_report
    report=write_report(data,best,output.with_suffix(".html"))
    print(json.dumps({"route":str(output),"report":str(report),"frames":best["frames"],
                      "predicted_berries":best["predicted"]["berries"],
                      "predicted_rooms":best["predicted"]["rooms"],
                      "runtime_verified":False},ensure_ascii=False),flush=True)
    return 0


def catalog(args):
    from .physics import Simulator
    files = discover_maps(args.maps)
    reports=[]
    for item in files:
        data=load_map(item)
        inventory=data.to_dict()
        inventory["simulation_capabilities"]=Simulator(data).capability_report()
        reports.append(inventory)
        print(f"{data.package}: {inventory['playable_room_count']} rooms, {inventory['strawberry_categories']}")
    write_json(args.output.resolve(),{"schema":"celeste-offline-catalog-v1","maps":reports,
                                      "runtime_verified":False,"sources":SOURCES})
    print(f"Catalog: {args.output.resolve()}")
    return 0


def verify(args):
    from .physics import Simulator
    route=load_route(args.route)
    if route.get("simulator_sha256")!=map_hash(Path(__file__).with_name("physics.py")):
        raise ValueError("Simulator changed since this route was generated")
    source=args.map or Path(route["map"]["path"])
    if map_hash(source)!=route["map"]["sha256"]:
        raise ValueError("Map contents differ from the planned route")
    state,_,rooms,cells=trace_route(Simulator(load_map(source)),route["start_room"],decode_actions(route["actions"]))
    if snapshot(state)!=route["predicted"]["state"]:
        raise ValueError("Offline replay differs from the saved final state")
    if sorted(rooms)!=route["predicted"]["rooms"] or sorted(state.berries)!=route["predicted"]["berries"]:
        raise ValueError("Offline replay achievements differ")
    print(f"Offline replay passed: {route['frames']} frames, {len(state.berries)} predicted berries. No game was started.")
    return 0


def batch(args):
    from .physics import Simulator
    jobs=[]
    for path in discover_maps(args.maps):
        if not args.all_sides and ("H-" in path.stem or "X-" in path.stem or path.stem=="8-Epilogue"):
            continue
        data=load_map(path)
        jobs.append({"map":str(path.resolve()),"room":data.start_room().name,
                     "seconds":args.seconds,"expansions":args.expansions,"max_goals":args.max_goals,
                     "target":None,"prefix":[],"allow_unsupported":args.allow_unsupported,"variant":0})
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(_worker,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                route=future.result()
                path=args.output/(Path(job["map"]).stem+".json")
                write_json(path,route)
                from .report import write_report
                write_report(load_map(Path(job["map"])),route,path.with_suffix(".html"))
                entry={"map":job["map"],"route":str(path.resolve()),"frames":route["frames"],
                       "predicted_berries":len(route["predicted"]["berries"]),"runtime_verified":False}
            except Exception as error:
                entry={"map":job["map"],"error":str(error)}
            manifest.append(entry)
            write_json(args.output/"batch.json",manifest)
            print(json.dumps(entry,ensure_ascii=False),flush=True)
    return 1 if any("error" in e for e in manifest) else 0


def positive(value):
    value=float(value)
    if not 0 < value < 86400:
        raise argparse.ArgumentTypeError("must be greater than zero and below 86400")
    return value


def count(value):
    result=int(value)
    if result<1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_count(value):
    result=int(value)
    if result<0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return result


def parser():
    from .assist import default_assist_workers
    p=argparse.ArgumentParser(description="Celeste 辅助机：手动游玩，点选地图目标后按需搜索")
    commands=p.add_subparsers(dest="command",required=True)
    assist=commands.add_parser("assist",help="open the selectable map window; physical controls remain active by default")
    assist.add_argument("--game-dir",type=Path,default=default_game_dir())
    assist.add_argument("--port",type=int,default=25102)
    assist.add_argument("--workers",type=count,default=default_assist_workers())
    assist.add_argument("--plan-seconds",type=positive,default=8)
    assist.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"assist-session.json")
    assist.add_argument("--launch",action="store_true",help="start normal Celeste with the assist bridge, without selecting or restarting a chapter")
    assist.add_argument("--fna3d-driver",help="optional FNA3D_FORCE_DRIVER override for a launched game")
    assist.add_argument("--sdl-videodriver",help="optional SDL_VIDEODRIVER override for a launched game")
    def run_assist(args):
        if not 1024<=args.port<=65535:
            raise ValueError("Port must be in 1024..65535")
        args.game_dir=normalize_game_dir(args.game_dir)
        from .assist_gui import run_assist_gui
        return run_assist_gui(args)
    assist.set_defaults(func=run_assist)
    c=commands.add_parser("catalog",help="read installed .bin files and inventory all rooms/berries")
    c.add_argument("--maps",type=Path,default=DEFAULT_MAPS)
    c.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"catalog.json")
    c.set_defaults(func=catalog)
    status=commands.add_parser("status",help="read actual progress and flag stalled searches without controlling the game")
    status.add_argument("--journal",type=Path,default=DEFAULT_OUTPUT/"continuous-live.json")
    def run_status(args):
        from .health import inspect_progress
        journal=json.loads(args.journal.read_text(encoding="utf-8"))
        print(json.dumps(inspect_progress(journal,load_map(Path(journal["map"]))),ensure_ascii=False,indent=2))
        return 0
    status.set_defaults(func=run_status)
    for name,handler in (("plan",plan),("batch",batch)):
        sub=commands.add_parser(name,help="offline search for a route" if name=="plan" else "offline batch search over chapters")
        sub.add_argument("--seconds",type=positive,default=30)
        sub.add_argument("--expansions",type=count,default=5000,help="maximum graph expansions per goal")
        sub.add_argument("--workers",type=count,default=min(4,os.cpu_count() or 1))
        sub.add_argument("--max-goals",type=count,default=32)
        sub.add_argument("--allow-unsupported",action="store_true",help="experimental predictions across unmodeled entities; playback refuses them")
        if name=="plan":
            sub.add_argument("--map",type=Path,default=DEFAULT_MAPS/"1-ForsakenCity.bin")
            sub.add_argument("--room")
            sub.add_argument("--target",help="room:NAME / berry:ROOM:ID / point:WORLD_X:WORLD_Y")
            sub.add_argument("--resume",type=Path)
            sub.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"route.json")
        else:
            sub.add_argument("--maps",type=Path,default=DEFAULT_MAPS)
            sub.add_argument("--all-sides",action="store_true")
            sub.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"batch")
        sub.set_defaults(func=handler)
    v=commands.add_parser("verify",help="recompute a saved route in the offline simulator")
    v.add_argument("route",type=Path)
    v.add_argument("--map",type=Path)
    v.set_defaults(func=verify)
    play=commands.add_parser("play",help="execute ONE previously planned route; never performs search")
    play.add_argument("route",type=Path)
    play.add_argument("--game-dir",type=Path,default=ROOT/"GAME"/"Celeste-1")
    play.add_argument("--launch",action="store_true",help="launch the selected patched game for this playback")
    play.add_argument("--port",type=int,default=25101)
    play.add_argument("--tolerance",type=positive,default=16)
    play.add_argument("--connect-timeout",type=positive,default=60)
    play.add_argument("--output",type=Path)
    play.add_argument("--install-bridge",action="store_true",help="back up and deploy the compiled telemetry DLL before launch")
    def run_play(args):
        from .playback import play_route
        return play_route(args)
    play.set_defaults(func=run_play)
    auto=commands.add_parser("run",help="live play: background search without pausing; native room respawns")
    auto.add_argument("--map",type=Path,default=DEFAULT_MAPS/"1-ForsakenCity.bin")
    auto.add_argument("--game-dir",type=Path,default=ROOT/"GAME"/"Celeste-1")
    auto.add_argument("--workers",type=count,default=min(4,os.cpu_count() or 1))
    auto.add_argument("--plan-seconds",type=positive,default=2)
    auto.add_argument("--expansions",type=count,default=5000)
    auto.add_argument("--retries",type=count,default=6,help="failed offline batches before a native current-room retry")
    auto.add_argument("--max-rollbacks",type=nonnegative_count,default=0,help="current-room retry limit; 0 keeps playing")
    auto.add_argument("--max-seconds",type=float,default=0,help="wall-clock run budget; 0 means continuous until chapter completion or user detach")
    auto.add_argument("--tolerance",type=positive,default=12)
    auto.add_argument("--port",type=int,default=25101)
    auto.add_argument("--connect-timeout",type=positive,default=60)
    auto.add_argument("--launch",action="store_true")
    auto.add_argument("--install-bridge",action="store_true")
    auto.add_argument("--resume",type=Path,help="reuse saved route rejection/search knowledge; does not replay chapters")
    auto.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"autoplay.json")
    def run_auto(args):
        if args.max_seconds<0 or not math.isfinite(args.max_seconds):
            raise ValueError("--max-seconds must be finite and nonnegative")
        if args.install_bridge and not args.launch:
            raise ValueError("--install-bridge requires --launch")
        from .live import run_live
        return run_live(args)
    auto.set_defaults(func=run_auto)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Stopped. No automatic game retry will be made.",file=sys.stderr)
        return 130
    except (OSError,ValueError,KeyError) as error:
        print(f"Error: {error}",file=sys.stderr)
        return 2
