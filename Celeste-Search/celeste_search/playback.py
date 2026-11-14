"""Execute exactly one precomputed route. This module never invokes the planner."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import socket
import struct
import sys
import time
import xml.etree.ElementTree as ET

from .maps import load_map
from .physics import Simulator
from .routes import decode_actions, load_route, map_hash, snapshot, write_json
from .game_platform import (MAC_BRIDGE_FILES, normalize_game_dir,
                            require_isolated_mac_game, require_game_stopped, launch_game_process)

MAX_MESSAGE = 1024*1024


def recv_exact(sock, size):
    result=bytearray()
    while len(result)<size:
        chunk=sock.recv(size-len(result))
        if not chunk:
            raise ConnectionError("Game disconnected during route playback")
        result.extend(chunk)
    return bytes(result)


def read_message(sock):
    tag=recv_exact(sock,4)
    if tag==b"STAT":
        if recv_exact(sock,1)!=b"E":
            raise ValueError("Invalid STATE marker")
        kind="state"
    elif tag==b"DONE":
        kind="done"
    else:
        raise ValueError(f"Expected SearchMode STATE/DONE; got {tag!r}. Install the new bridge.")
    length=struct.unpack("<I",recv_exact(sock,4))[0]
    if not 0<length<=MAX_MESSAGE:
        raise ValueError("Invalid playback JSON length")
    data=json.loads(recv_exact(sock,length))
    if data.get("schema_version")!="celeste-playback-state-v1":
        raise ValueError("Unexpected playback state schema")
    return kind,data


def session_xml(route, seconds):
    # Celeste numeric AreaKey differs from content filenames after the epilogue.
    stem=Path(route["map"]["path"]).stem
    area_names={"Intro":0,"ForsakenCity":1,"OldSite":2,"CelestialResort":3,
                "GoldenRidge":4,"MirrorTemple":5,"Reflection":6,"Summit":7,
                "Epilogue":8,"Core":9}
    if stem=="LostLevels":
        area,mode=10,0
    else:
        prefix,name=stem.split("-",1)
        area=area_names[name]
        mode=1 if prefix.endswith("H") else 2 if prefix.endswith("X") else 0
    root=ET.Element("SessionParameters")
    values={"SearchMode":"true","EndOnDeath":"true","EndOnLevelExit":"false",
            "TimeoutSeconds":seconds,"AreaKey":area,"AreaMode":mode,"Level":route["start_room"],
            "ObjectiveXCoordinate":route["predicted"]["state"]["x"],
            "ObjectiveYCoordinate":route["predicted"]["state"]["y"]}
    for name,value in values.items():
        ET.SubElement(root,name).text=str(value)
    return ET.tostring(root,encoding="utf-8")


def install_bridge(game_dir:Path, bridge_dir=None):
    workspace=Path(__file__).resolve().parents[2]
    game_dir=require_isolated_mac_game(game_dir)
    require_game_stopped(game_dir)
    if bridge_dir is None:
        bridge_dir=(workspace/"artifacts/bridge-mac" if sys.platform=="darwin" else
                    workspace/"Celeste-Instrumentation/CelesteInstrumentation/CelesteInstrumentation/bin/Release/net4.5")
    bridge_dir=Path(bridge_dir)
    names=MAC_BRIDGE_FILES if sys.platform=="darwin" else ("CelesteInstrumentation.dll",)
    sources=[bridge_dir/name for name in names]
    for compiled in sources:
        if not compiled.is_file():
            raise FileNotFoundError(f"Build the platform bridge first; missing {compiled}")
    for compiled in sources:
        destination=game_dir/compiled.name
        if destination.is_file() and destination.read_bytes()==compiled.read_bytes():
            continue
        if destination.is_file():
            backup=destination.with_name(destination.name+".before-search-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            shutil.copy2(destination,backup)
        shutil.copy2(compiled,destination)


def configure_game(game_dir,port):
    path=normalize_game_dir(game_dir)/"InstrumentationParameters.xml"
    original=path.read_bytes() if path.is_file() else None
    root=ET.fromstring(original) if original is not None else ET.Element("GlobalParameters")
    for key,value in {"Enabled":"true","DebugRendering":"false","RouteRendering":"false",
                      "Headless":"false","GamespeedMultiplier":"1","ConnectionIP":"127.0.0.1",
                      "ConnectionPort":str(port)}.items():
        item=root.find(key)
        if item is None:
            item=ET.SubElement(root,key)
        item.text=value
    path.write_bytes(ET.tostring(root,encoding="utf-8",xml_declaration=True))
    return path,original


def restore_configuration(configuration):
    path,original=configuration
    if original is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(original)


def distance_to_prediction(observed,expected,index,lag=0):
    """Compare at the documented pre-input boundary; never shift the route."""
    distances=[math.hypot(observed["x"]-state.x,observed["y"]-state.y)
               for state in expected[max(0,index-lag):min(len(expected),index+1)]
               if state.room==observed["room"]]
    return min(distances,default=math.inf)


def play_route(args):
    route=load_route(args.route)
    if route.get("simulator_sha256")!=map_hash(Path(__file__).with_name("physics.py")):
        raise ValueError("Simulator changed since planning; recompute this route")
    actions=decode_actions(route["actions"])
    if not actions:
        raise ValueError("Empty prediction: no route to execute")
    if route.get("allow_unsupported"):
        raise ValueError("Experimental routes with unsupported mechanics cannot be played automatically")
    game_dir=normalize_game_dir(args.game_dir)
    source=game_dir/"Content"/"Maps"/Path(route["map"]["path"]).name
    if map_hash(source)!=route["map"]["sha256"]:
        raise ValueError("Selected game's map differs from planned map")
    simulator=Simulator(load_map(source))
    state=simulator.initial_state(route["start_room"])
    expected=[state]
    for action in actions:
        state=simulator.step(state,action)
        if state.dead:
            raise ValueError("Route fails offline replay; recompute before playback")
        expected.append(state)
    if snapshot(state)!=route["predicted"]["state"]:
        raise ValueError("Simulator changed since planning; recompute route before playback")
    for name in {s.room for s in expected}:
        if not simulator.capability_report(name)["planning_supported"]:
            raise ValueError(f"Route enters room {name} with unsupported mechanics")
    if not 1024<=args.port<=65535:
        raise ValueError("Port must be in 1024..65535")
    if args.install_bridge and not args.launch:
        raise ValueError("--install-bridge requires --launch so the selected copy is configured")
    if args.launch and not (game_dir/"Celeste.exe").is_file():
        raise FileNotFoundError(game_dir/"Celeste.exe")
    output=(args.output or args.route.with_name(args.route.stem+".playback.json")).resolve()
    result={"schema":"celeste-route-playback-v1","route":str(args.route.resolve()),
            "verified":False,"planned_frames":len(actions),"sent_frames":0,
            "observed_rooms":[],"confirmed_berries":[],"observations":[],
            "reason":"not_started","automatic_retry":False}
    configuration=None
    process=None
    observed_rooms=set()
    confirmed=set()
    try:
        with socket.socket() as listener:
            # Windows exclusive address ownership avoids colliding with old training clients.
            listener.setsockopt(socket.SOL_SOCKET,getattr(socket,"SO_EXCLUSIVEADDRUSE",socket.SO_REUSEADDR),1)
            listener.bind(("127.0.0.1",args.port))
            listener.listen(1)
            listener.settimeout(args.connect_timeout)
            if args.launch:
                if args.install_bridge:
                    install_bridge(game_dir)
                configuration=configure_game(game_dir,args.port)
                process=launch_game_process(game_dir)
            print(f"Waiting for patched Celeste on 127.0.0.1:{args.port}; one execution only",flush=True)
            connection,_=listener.accept()
            with connection:
                connection.settimeout(20)
                connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                if recv_exact(connection,8)!=b"LOADINFO":
                    raise ValueError("Unexpected initial handshake")
                xml=session_xml(route,max(30,len(actions)/60+10)).replace(b"<EndOnDeath>true</EndOnDeath>",b"<EndOnDeath>false</EndOnDeath>")
                connection.sendall(xml)
                stop_reason=None
                previous_frame=-1
                while True:
                    kind,observed=read_message(connection)
                    observed_rooms.add(observed["room"])
                    confirmed.update(observed["berries"])
                    result["observations"].append(observed)
                    if kind=="done":
                        connection.sendall(b"\x00")
                        result["terminal"]=observed
                        result["reason"]=stop_reason or observed.get("reason","ended")
                        break
                    frame=observed["frame"]
                    if observed.get("input_phase")!="before_virtual_inputs":
                        stop_reason="incompatible_input_phase"
                    if type(frame) is not int or frame!=previous_frame+1:
                        stop_reason="invalid_frame_sequence"
                    previous_frame=frame
                    if not stop_reason:
                        distance=distance_to_prediction(observed,expected,min(frame,len(actions)))
                        if distance>args.tolerance:
                            stop_reason="prediction_diverged"
                            result["divergence"]={"frame":frame,"distance_pixels":distance if math.isfinite(distance) else None,
                                                  "observed":observed,"expected":snapshot(expected[min(frame,len(actions))])}
                    if stop_reason or frame>=len(actions):
                        stop_reason=stop_reason or "route_finished"
                        connection.sendall(b"\xff")
                    elif frame<len(actions):
                        connection.sendall(bytes([actions[frame]]))
                        result["sent_frames"]+=1
                    if stop_reason=="prediction_diverged":
                        print(f"Stopped at frame {frame}: offline prediction differs from game; no retry",flush=True)
        expected_berries=set(route["predicted"]["berries"])
        expected_rooms=set(route["predicted"]["rooms"])
        result["verified"]=(result["reason"] in {"route_finished","chapter_complete"}
                            and result["sent_frames"]==len(actions)
                            and expected_berries<=confirmed and expected_rooms<=observed_rooms)
        if result["reason"]=="route_finished" and not result["verified"]:
            result["reason"]="predicted_achievements_not_observed"
    except (OSError,ValueError,KeyError) as error:
        result["reason"]="playback_error"
        result["error"]=str(error)
    finally:
        # Live-mode detach restores human input and leaves the current game open.
        if configuration:
            restore_configuration(configuration)
        result["observed_rooms"]=sorted(observed_rooms)
        result["confirmed_berries"]=sorted(confirmed)
        write_json(output,result)
    print(json.dumps({"result":str(output),"verified":result["verified"],"reason":result["reason"],
                      "confirmed_berries":result["confirmed_berries"]},ensure_ascii=False),flush=True)
    return 0 if result["verified"] else 3
