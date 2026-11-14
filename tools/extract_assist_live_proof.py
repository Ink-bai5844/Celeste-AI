"""Read-only live-route audit by explicit trace session and request id.

Indexes byte spans without decoding old STATE payloads. Replays recorded inputs
from the first actual execution STATE using current physics; never opens a game
socket, starts the game, or sends inputs. Model parity and live success are
reported separately, including a two-second post-success observation window.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Celeste-Search"))
SESSION_PATTERN=re.compile(rb'"trace_session"\s*:\s*"([^"\\]+)"')
TIMER_FIELDS=("grace","jump_buffer","dash_buffer","var_jump","dash_frames","dash_attack",
              "dash_cooldown","refill_cooldown","force_frames","wall_slide","wall_retention",
              "climb_wait","auto_jump","collect_delay","collect_ground","lift_timer")
MOTION_FIELDS=("x","y","vx","vy","stamina","rem_x","rem_y","lift_vx","lift_vy",
               "var_speed","wall_retained","hop_wait_x_speed")
EXACT_FIELDS=("room","state","dashes","ducking","dead","berries","following","follow_order")


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(path)


def _hash_region(stream,start,length):
    stream.seek(start)
    return sha256(stream.read(length)).hexdigest()


def index_trace(path,index_path):
    """Incremental append index, guarded against replacement/truncation/rewrites."""
    path,index_path=Path(path).resolve(),Path(index_path)
    stat=path.stat()
    try:
        cached=json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError,ValueError):
        cached={}
    with path.open("rb") as stream:
        valid=(cached.get("path")==str(path) and cached.get("inode")==stat.st_ino
               and 0<=cached.get("indexed_bytes",-1)<=stat.st_size)
        if valid:
            valid=(_hash_region(stream,0,cached["head_bytes"])==cached["head_sha256"] and
                   _hash_region(stream,cached["indexed_bytes"]-cached["tail_bytes"],cached["tail_bytes"])==cached["tail_sha256"])
        index=cached if valid else dict(path=str(path),inode=stat.st_ino,indexed_bytes=0,line_count=0,sessions={})
        offset=index["indexed_bytes"]
        line_number=index["line_count"]
        stream.seek(offset)
        scanned_from=offset
        while offset<stat.st_size:
            start=offset
            line=stream.readline()
            if not line or not line.endswith(b"\n") or start+len(line)>stat.st_size:
                break
            offset+=len(line)
            line_number+=1
            match=SESSION_PATTERN.search(line)
            if not match:
                continue
            session=match.group(1).decode("utf-8")
            span=index["sessions"].setdefault(session,dict(first_byte=start,first_line=line_number))
            span.update(last_end=offset,last_line=line_number)
        index.update(indexed_bytes=offset,line_count=line_number,head_bytes=min(4096,offset),tail_bytes=min(4096,offset))
        index["head_sha256"]=_hash_region(stream,0,index["head_bytes"])
        index["tail_sha256"]=_hash_region(stream,offset-index["tail_bytes"],index["tail_bytes"])
    write_json(index_path,index)
    return index,offset-scanned_from


def timestamp(record):
    return record.get("time",record.get("wall_time_ns",0)/1e9)


def read_request_records(path,index,session,request_id,tail_seconds):
    span=index["sessions"].get(session)
    if span is None:
        raise ValueError(f"trace session {session} was not found")
    records=[]
    success_time=None
    parsed=0
    needle=session.encode("utf-8")
    with Path(path).open("rb") as stream:
        stream.seek(span["first_byte"])
        line_number=span["first_line"]-1
        while stream.tell()<span["last_end"]:
            offset=stream.tell()
            line=stream.readline()
            line_number+=1
            if needle not in line:
                continue
            try:
                record=json.loads(line)
            except ValueError:
                continue
            parsed+=1
            if record.get("trace_session")!=session:
                continue
            record.update(_line=line_number,_byte=offset)
            kind=record.get("kind")
            identifier=record.get("request_id",record.get("event",{}).get("request_id"))
            if kind=="state" and identifier==request_id and record.get("mode") in ("success","failed") and success_time is None:
                success_time=timestamp(record)
            if (identifier==request_id or success_time is not None and (kind=="state" or
                    kind=="event" and record.get("event",{}).get("event") in {"target_confirmed","route_confirmed"})
                    or kind in {"hello","harness_started","worker_warmup_completed","connection_error","startup_error","disconnected"}):
                records.append(record)
            # Keep one bounding STATE beyond the requested tail, but never
            # consume unrelated later sessions or an entire long idle log.
            if success_time is not None and kind=="state" and timestamp(record)>success_time+tail_seconds:
                if record not in records:
                    records.append(record)
                break
    return records,parsed,span


def observed_value(observation,field):
    if field in observation:
        return observation[field]
    return observation.get("model_state",{}).get(field)


def compact(record):
    observation=record["observation"]
    return dict(source_line=record["_line"],source_byte=record["_byte"],time=timestamp(record),
                mode=record.get("mode"),action_sent=record.get("action"),
                route_index=record.get("route_index"),waypoint_index=record.get("waypoint_index"),
                **{field:observed_value(observation,field) for field in
                   ("frame","room","x","y","vx","vy","on_ground","dead","deaths","spawn_id",
                    "control_mode","human_input_mask","human_menu_input","human_input_epoch","berries","following")})


def compare_model(model,observation,tolerance,frame_index,summary):
    unavailable=set(observation.get("model_state_unavailable",()))
    for field in MOTION_FIELDS+TIMER_FIELDS+EXACT_FIELDS:
        actual=observed_value(observation,field)
        if field in unavailable or actual is None:
            summary["unavailable_fields"][field]+=1
            continue
        predicted=getattr(model,field)
        if field in MOTION_FIELDS+TIMER_FIELDS:
            error=abs(float(predicted)-float(actual))
            summary["max_absolute_error"][field]=max(summary["max_absolute_error"].get(field,0),error)
            mismatch=not math.isfinite(error) or error>tolerance
        else:
            if field in ("berries","following"):
                predicted,actual=sorted(predicted),sorted(actual)
            elif field=="follow_order":
                predicted,actual=list(predicted),list(actual)
            mismatch=predicted!=actual
        if mismatch:
            summary["difference_count_by_field"][field]+=1
            if len(summary["first_differences"])<24:
                summary["first_differences"].append(dict(action_index=frame_index,field=field,predicted=predicted,observed=actual))


def post_success_tail(success,records,tail_seconds):
    """Attribute only the observed neutral interval before manual input/new work."""
    final=success["observation"]
    neutral=[]
    human=None
    next_request=None
    for record in records:
        if timestamp(record)>timestamp(success)+tail_seconds:
            break
        if record.get("request_id")!=success.get("request_id"):
            next_request=record
            break
        observation=record["observation"]
        if (observation.get("human_input_mask") or observation.get("human_menu_input")
                or observation.get("human_input_epoch")!=final.get("human_input_epoch")):
            human=record
            break
        neutral.append(record)
    seconds=timestamp(neutral[-1])-timestamp(success) if neutral else 0
    safe=bool(neutral) and all(not r["observation"].get("dead")
        and r["observation"].get("deaths")==final.get("deaths")
        and r["observation"].get("spawn_id")==final.get("spawn_id") for r in neutral)
    contiguous=bool(neutral) and all(b["observation"]["frame"]==a["observation"]["frame"]+1
        for a,b in zip([success]+neutral,neutral))
    coverage=seconds>=tail_seconds-1/30
    return dict(requested_seconds=tail_seconds,observed_neutral_seconds=round(seconds,4),
        neutral_observations=len(neutral),coverage_complete_within_one_frame=coverage,
        frames_contiguous=contiguous,full_neutral_tail_verified=safe and contiguous and coverage,
        interrupted_by_human=human is not None,first_human_input=compact(human) if human else None,
        interrupted_by_new_request=next_request is not None,
        next_request_id=next_request.get("request_id") if next_request else None,
        safe_before_human=safe,manual_input_and_later_requests_excluded=True,
        last_neutral_observation=compact(neutral[-1]) if neutral else None)


def request_connection_audit(records,request_id,plan,success,tail_seconds):
    """Attribute errors to this request, keeping earlier session faults visible.

    Start at its first confirmation/planning record (or plan if unavailable).
    An untagged disconnect inside the interval still counts. Tail attribution
    stops at a new request or human input, matching the neutral-tail boundary.
    """
    request_records=[record for record in records if
        record.get("request_id",record.get("event",{}).get("request_id"))==request_id]
    start_line=min([plan["_line"]]+[record["_line"] for record in request_records])
    end_time=timestamp(success)+tail_seconds
    end_line=math.inf
    for record in records:
        if record["_line"]<=success["_line"]:
            continue
        identifier=record.get("request_id",record.get("event",{}).get("request_id"))
        if record.get("kind")=="event" and record.get("event",{}).get("event") in {"target_confirmed","route_confirmed"}:
            interrupted=identifier!=request_id
        elif record.get("kind")=="state":
            observation=record["observation"]
            interrupted=(identifier!=request_id or observation.get("human_input_mask")
                or observation.get("human_menu_input") or observation.get("human_input_epoch")!=
                success["observation"].get("human_input_epoch"))
        else:
            interrupted=False
        if interrupted:
            end_line=record["_line"]
            break
    prior,execution,tail=[],[],[]
    for record in records:
        if record.get("kind") not in {"connection_error","startup_error","disconnected"}:
            continue
        item=dict(source_line=record["_line"],kind=record["kind"],time=timestamp(record),
                  request_id=record.get("request_id"),error=record.get("error"))
        if record["_line"]<start_line:
            prior.append(item)
        elif record["_line"]<=success["_line"]:
            execution.append(item)
        elif record["_line"]<end_line and timestamp(record)<=end_time:
            tail.append(item)
    return dict(request_start_source_line=start_line,
        session_prior_connection_errors=prior,
        session_prior_connection_error_counts=dict(Counter(item["kind"] for item in prior)),
        connection_errors_through_success=execution,connection_errors_post_success_tail=tail)


def audit_failed_execution(proof,records,plan_record,states,*,map_path,tolerance):
    """Retain the actual action prefix and first divergence even without success."""
    from celeste_search.maps import load_map
    from celeste_search.physics import Simulator
    from celeste_search.runtime_state import state_from_observation
    execution=[record for record in states if record.get("mode")=="executing"]
    if not execution:
        proof.update(reason="execution_has_no_observed_success",plan_source_line=plan_record["_line"],executing_states=0)
        return proof
    terminal=next((record for record in states if record["_line"]>execution[-1]["_line"]),None)
    aligned=execution+([terminal] if terminal else [])
    plan=plan_record["candidate"]
    hello=next((record for record in records if record.get("kind")=="hello"),{})
    selected_map=Path(map_path or hello.get("map_path","")).resolve()
    if not selected_map.is_file():
        raise ValueError("map path was not recorded; pass --map-path")
    simulator=Simulator(load_map(selected_map))
    model=state_from_observation(simulator.map,execution[0]["observation"])
    parity=dict(tolerance=tolerance,max_absolute_error={},difference_count_by_field=Counter(),
                unavailable_fields=Counter(),first_differences=[])
    frames=[]
    first_position_difference=None
    for position,record in enumerate(aligned):
        if position:
            model=simulator.step(model,execution[position-1]["action"])
        observation=record["observation"]
        compare_model(model,observation,tolerance,position,parity)
        details=dict(action_index=position,source_line=record["_line"],time=timestamp(record),
            mode=record.get("mode"),action_sent=record.get("action"),route_index=record.get("route_index"),
            observation={key:value for key,value in observation.items() if key not in ("room_entities","capabilities","state_fingerprint")},
            independent_prediction=model.to_dict(),original_prediction=plan["expected"][position] if position<len(plan["expected"]) else None)
        frames.append(details)
        if first_position_difference is None and (abs(model.x-observation["x"])>tolerance or abs(model.y-observation["y"])>tolerance):
            first_position_difference=position
    after=[record for record in states if terminal and record["_line"]>terminal["_line"]]
    initial=execution[0]["observation"]
    first_death=next((record for record in after if record["observation"].get("dead") or record["observation"].get("deaths")!=initial.get("deaths")),None)
    first_respawn=next((record for record in after if record["observation"].get("spawn_id")!=initial.get("spawn_id")),None)
    proof.update(reason="execution_has_no_observed_success",source_plan_line=plan_record["_line"],
        strategy=plan.get("strategy"),route=plan["route"],planned_actions=plan["actions"],
        actual_action_prefix=[record["action"] for record in execution],executed_action_count=len(execution),
        prefix_matches_plan=[record["action"] for record in execution]==plan["actions"][:len(execution)],
        execution_checks=dict(alive_until_release=all(not r["observation"].get("dead") for r in aligned),
            human_masks_zero=all(r["observation"].get("human_input_mask")==0 and r["observation"].get("human_menu_input") is False for r in aligned),
            human_epoch_unchanged=len({r["observation"].get("human_input_epoch") for r in aligned})==1,
            deaths_unchanged_until_release=len({r["observation"].get("deaths") for r in aligned})==1,
            spawn_unchanged_until_release=len({r["observation"].get("spawn_id") for r in aligned})==1,
            frames_contiguous=all(b["observation"]["frame"]==a["observation"]["frame"]+1 for a,b in zip(aligned,aligned[1:])),
            actions_released_after_failure=terminal is not None and terminal.get("action")==128 and all(r.get("action")==128 for r in after)),
        first_actual_observation=execution[0]["observation"],plan_anchor=plan["anchor_state"],
        map_path=str(selected_map),physics_sha256=sha256((ROOT/"Celeste-Search/celeste_search/physics.py").read_bytes()).hexdigest(),
        first_difference=parity["first_differences"][0] if parity["first_differences"] else None,
        first_position_difference_index=first_position_difference,comparison=parity,execution_frames=frames,
        terminal=compact(terminal) if terminal else None,
        first_subsequent_death=compact(first_death) if first_death else None,
        first_subsequent_respawn=compact(first_respawn) if first_respawn else None,
        after_release=[compact(record) for record in after[:125]],
        events=[dict(source_line=record["_line"],time=timestamp(record),**record["event"]) for record in records if record.get("kind")=="event"])
    return proof


def audit(trace,session,request_id,*,index_path=None,map_path=None,tolerance=.0001,tail_seconds=2):
    from celeste_search.assist_routes import start_route_progress,advance_route_progress,route_complete
    from celeste_search.maps import load_map
    from celeste_search.physics import Simulator
    from celeste_search.runtime_state import state_from_observation,falling_snapshot_from_observation,RuntimeStateError

    trace=Path(trace).resolve()
    index_path=Path(index_path) if index_path else trace.with_name(trace.stem+"-session-index.json")
    index,indexed_bytes=index_trace(trace,index_path)
    records,parsed,span=read_request_records(trace,index,session,request_id,tail_seconds)
    proof=dict(schema="celeste-live-proof-v2",trace_session=session,request_id=request_id,
               source_trace=str(trace),source_session_span=span,source_index=str(index_path),
               newly_indexed_bytes=indexed_bytes,session_records_decoded=parsed,game_inputs_sent_by_audit=0,
               created_unix_seconds=time.time(),runtime_route_success_verified=False)
    plans=[record for record in records if record.get("kind")=="plan" and record.get("request_id")==request_id]
    if len(plans)!=1:
        proof["reason"]="no_executable_plan" if not plans else "multiple_executions_require_separate_audit"
        proof["plan_source_lines"]=[record["_line"] for record in plans]
        proof["events"]=[dict(source_line=record["_line"],**record["event"]) for record in records if record.get("kind")=="event"]
        return proof
    plan_record=plans[0]
    plan=plan_record["candidate"]
    states=[record for record in records if record.get("kind")=="state" and record.get("request_id")==request_id and record["_line"]>plan_record["_line"]]
    execution=[record for record in states if record.get("mode")=="executing"]
    success=next((record for record in states if record.get("mode")=="success"),None)
    if not execution or success is None:
        return audit_failed_execution(proof,records,plan_record,states,map_path=map_path,tolerance=tolerance)
    aligned=execution+[success]
    actions=plan["actions"]
    route=plan["route"]
    expected=plan["expected"]
    observations=[record["observation"] for record in aligned]
    count_matches=len(execution)==len(actions) and len(expected)==len(aligned)
    if not count_matches:
        proof.update(reason="incomplete_execution_trace",action_count=len(actions),expected_count=len(expected),observed_count=len(aligned))
        return proof
    events=[record for record in records if record.get("kind")=="event" and record["event"].get("request_id")==request_id]
    search=next((record for record in events if record["event"]["event"]=="background_target_search"),None)
    execute=next((record for record in events if record["event"]["event"]=="execute_target_route"),aligned[0])
    hello=next((record for record in records if record.get("kind")=="hello"),{})
    selected_map=Path(map_path or hello.get("map_path","")).resolve()
    if not selected_map.is_file():
        raise ValueError("map path was not recorded; pass --map-path")
    data=load_map(selected_map)
    simulator=Simulator(data)
    model=state_from_observation(data,observations[0])
    model_start=model.to_dict()
    model_progress=start_route_progress(route,model)
    actual_progress=start_route_progress(route,observations[0])
    actual_hits=[0]*actual_progress.index
    model_hits=[0]*model_progress.index
    parity=dict(tolerance=tolerance,timer_units="player-update frame counts exported by model_state",
                motion_fields=list(MOTION_FIELDS),timer_fields=list(TIMER_FIELDS),exact_fields=list(EXACT_FIELDS),
                max_absolute_error={},difference_count_by_field=Counter(),unavailable_fields=Counter(),first_differences=[])
    plan_comparison=dict(frame_offset=observations[0]["frame"]-expected[0]["frame"],max_absolute_error={},difference_count_by_field=Counter(),
        mechanism_comparison=dict(scope="Exact native FallingBlock six-tuples after Single decoding; other mechanism kinds are explicitly unverified",
                                  observation_comparisons=0,tuple_comparisons=0,unavailable_observations=0,unverified_records_skipped=0))
    replay_alive=not model.dead
    for position,(observation,prediction) in enumerate(zip(observations,expected)):
        if position:
            before=model
            model=simulator.step(model,actions[position-1])
            old=model_progress.index
            model_progress=advance_route_progress(route,model_progress,before,model,actions[position-1])
            model_hits.extend([position]*(model_progress.index-old))
            old=actual_progress.index
            actual_progress=advance_route_progress(route,actual_progress,observations[position-1],observation,actions[position-1])
            actual_hits.extend([position]*(actual_progress.index-old))
        replay_alive=replay_alive and not model.dead
        compare_model(model,observation,tolerance,position,parity)
        for field,value in prediction.items():
            if field=="mechanisms":
                # Mechanisms live in room_entities, not a top-level STATE key.
                # Use the same verified native decoder as the strict execution
                # guard instead of comparing the prediction with missing None.
                room=data.rooms.get(prediction.get("room"))
                identities={f"{room.name}:{entity.id}" for entity in room.entities
                            if entity.kind=="fallingBlock"} if room else set()
                verified=[list(record) for record in value if record[0] in identities]
                audit=plan_comparison["mechanism_comparison"]
                audit["unverified_records_skipped"]+=len(value)-len(verified)
                if not identities:
                    continue
                try:
                    actual=falling_snapshot_from_observation(data,observation)
                except RuntimeStateError:
                    audit["unavailable_observations"]+=1
                    continue
                audit["observation_comparisons"]+=1
                audit["tuple_comparisons"]+=len(verified)
                if actual!=sorted(verified,key=lambda record:record[0]):
                    plan_comparison["difference_count_by_field"][field]+=1
                continue
            actual=observation.get(field)
            if field=="frame":
                actual-=plan_comparison["frame_offset"]
            if type(value) in (float,int) and type(actual) in (float,int):
                error=abs(value-actual)
                plan_comparison["max_absolute_error"][field]=max(plan_comparison["max_absolute_error"].get(field,0),error)
                mismatch=error>tolerance
            else:
                mismatch=(sorted(value)!=sorted(actual) if field in ("berries","following") else value!=actual)
            if mismatch:
                plan_comparison["difference_count_by_field"][field]+=1
    checks=dict(actions_match_plan=[record["action"] for record in execution]==actions,
                actual_frames_contiguous=all(b["frame"]==a["frame"]+1 for a,b in zip(observations,observations[1:])),
                input_phase_before_virtual_inputs=all(o.get("input_phase")=="before_virtual_inputs" for o in observations),
                alive_every_observation=all(not o.get("dead") for o in observations),
                deaths_unchanged=len({o.get("deaths") for o in observations})==1,
                spawn_unchanged=len({o.get("spawn_id") for o in observations})==1,
                no_human_input=all(o.get("human_input_mask")==0 and o.get("human_menu_input") is False for o in observations),
                human_epoch_unchanged=len({o.get("human_input_epoch") for o in observations})==1,
                ordered_route_complete=route_complete(route,actual_progress),
                controller_reported_complete=success.get("waypoint_index")==len(route["waypoints"]))
    tail_records=[record for record in records if record.get("kind")=="state" and record["_line"]>success["_line"]]
    first_manual=next((record for record in tail_records if record["observation"].get("control_mode")=="manual"),None)
    checks["manual_returned_next_observation"]=bool(first_manual and first_manual["observation"]["frame"]==observations[-1]["frame"]+1 and first_manual["action"]==128)
    tail=post_success_tail(success,tail_records,tail_seconds)
    connection_audit=request_connection_audit(records,request_id,plan_record,success,tail_seconds)
    runtime_success=all(checks.values()) and not connection_audit["connection_errors_through_success"]
    proof.update(runtime_route_success_verified=runtime_success,
                 safe_handoff_verified=runtime_success and tail["full_neutral_tail_verified"]
                     and not connection_audit["connection_errors_post_success_tail"],checks=checks,
                 route=route,public_waypoints=len(route["waypoints"]),actions=actions,action_frames=len(actions),
                 source_plan_line=plan_record["_line"],strategy=plan.get("strategy"),
                 actual_waypoint_indices=actual_hits,planned_waypoint_indices=plan.get("waypoint_indices"),
                 waypoint_hits=[dict(waypoint=number+1,action_index=position,observation=compact(aligned[position])) for number,position in enumerate(actual_hits)],
                 search_dispatch_to_execute_ms=round((timestamp(execute)-timestamp(search))*1000,4) if search else None,
                 execute_to_success_ms=round((timestamp(success)-timestamp(execute))*1000,4),
                 original_prediction_comparison=plan_comparison,
                 independent_replay=dict(map_path=str(selected_map),map_sha256=sha256(selected_map.read_bytes()).hexdigest(),
                     physics_sha256=sha256((ROOT/"Celeste-Search/celeste_search/physics.py").read_bytes()).hexdigest(),
                     origin="first actual executing STATE; no spawn substitution or position override",initial_state=model_start,
                     alive_every_frame=replay_alive,ordered_route_complete=route_complete(route,model_progress),
                     waypoint_indices=model_hits,full_exported_field_parity=not parity["difference_count_by_field"],comparison=parity),
                 initial=compact(aligned[0]),success=compact(success),manual_release=compact(first_manual) if first_manual else None,
                 post_success_tail=tail,**connection_audit)
    return proof


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,default=ROOT/"artifacts/live-assist/trace.jsonl")
    parser.add_argument("--session",required=True)
    parser.add_argument("--request-id",required=True,type=int)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--index",type=Path)
    parser.add_argument("--map-path",type=Path)
    parser.add_argument("--tolerance",type=float,default=.0001)
    parser.add_argument("--tail-seconds",type=float,default=2)
    args=parser.parse_args(argv)
    if not math.isfinite(args.tolerance) or args.tolerance<0 or not math.isfinite(args.tail_seconds) or args.tail_seconds<0:
        parser.error("tolerance and tail seconds must be finite and nonnegative")
    proof=audit(args.trace,args.session,args.request_id,index_path=args.index,map_path=args.map_path,
                tolerance=args.tolerance,tail_seconds=args.tail_seconds)
    write_json(args.output,proof)
    print(json.dumps(dict(output=str(args.output.resolve()),runtime_success=proof["runtime_route_success_verified"],
                         reason=proof.get("reason"),frames=proof.get("action_frames"),
                         executed_prefix_frames=proof.get("executed_action_count"),first_difference=proof.get("first_difference"),
                         decoded_records=proof["session_records_decoded"],newly_indexed_bytes=proof["newly_indexed_bytes"],
                         waypoint_indices=proof.get("actual_waypoint_indices"),
                         model_parity=proof.get("independent_replay",{}).get("full_exported_field_parity"),
                         model_differences=proof.get("independent_replay",{}).get("comparison",{}).get("difference_count_by_field"),
                         post_success_tail=proof.get("post_success_tail")),ensure_ascii=False))
    return 0 if proof["runtime_route_success_verified"] else 1


if __name__=="__main__":
    raise SystemExit(main())
