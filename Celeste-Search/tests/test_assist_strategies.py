"""Diverse named recipes retain the same strict whole-route execution contract."""
from dataclasses import dataclass,replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import (plan_to_route,route_strategy_specs,_OrderedFrontier,
                                           _wall_setup_macros,target_reached,_refill_guidance_route)
from celeste_search.assist_routes import normalize_route
from celeste_search.assist_validation import target_from_world
from celeste_search.assist_strategies import (ZipBoardGoal,_ride_zip,_support_candidates,
                                             TransientSupportGoal,_coast_support_arrivals,zip_opens_target)
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import Simulator,SimState
from celeste_search.planner import Goal,Heuristic,search_leg
from celeste_search.strategies import spring_goal


def fixture(entities=()):
    room=Room("a",0,0,192,112,("0"*24,)*12+("1"*24,)*2,tuple(entities),((16.,96.),))
    return MapData("portfolio",{"a":room},Path("portfolio.bin"))


def underhang_fixture():
    rows=[["0"]*30 for _ in range(28)]
    for y in range(19,28):
        for x in (15,16,21):
            rows[y][x]="1"
    for x in (*range(11,14),*range(16,19)):
        rows[22][x]="1"
    entities=(Entity("crumbleBlock",30,104,120,24,8),
              Entity("jumpThru",31,96,152,24,8),
              Entity("spikesUp",18,96,152,32,3),
              Entity("spring",34,144,176,0,0),
              Entity("jumpThru",36,152,176,16,8),
              Entity("spikesUp",5,56,176,56,3))
    room=Room("a",0,0,240,224,tuple("".join(row) for row in rows),entities,((160.,176.),))
    return MapData("underhang",{"a":room},Path("underhang.bin"))


class StrategyTests(unittest.TestCase):
    def test_portfolio_names_are_unique_and_have_distinct_algorithm_families(self):
        specs=route_strategy_specs()
        self.assertGreaterEqual(len(specs),20)
        self.assertEqual(len(specs),len({row["name"] for row in specs}))
        self.assertTrue({"direct","semantic","rested_support","support_graph","zip_graph","ordered"}
                        <= {row["family"] for row in specs})
        self.assertTrue(all(row["label"] for row in specs))
        specs[0]["name"]="mutated"
        self.assertNotEqual(route_strategy_specs()[0]["name"],"mutated")

    def test_constraints_and_ordered_points_keep_global_frontier(self):
        point={"room":"a","x":96,"y":88}
        for route in ({"waypoints":[point,point]},
                      {"waypoints":[point],"segments":[{"no_dash":True}]}):
            self.assertTrue(all(row["family"]=="ordered" for row in route_strategy_specs(route)))

    def test_spiked_zip_support_candidates_use_safe_sides_instead_of_the_top(self):
        entities=[Entity("zipMover",8,48,56,32,24,{},((104.,56.),)),
                  Entity("spikesUp",9,48,56,32,3)]
        data=fixture(entities);simulator=Simulator(data);start=simulator.initial_state("a")
        goal=Goal("target","a",144,72)
        for mode in ("forward","climb","underhang","zip"):
            candidates=_support_candidates(simulator,start,goal,mode,())
            self.assertFalse(any(isinstance(row[2],ZipBoardGoal) for row in candidates))
            grips=[row[2] for row in candidates if getattr(row[2],"identity","")=="a:8"]
            self.assertEqual({grip.facing for grip in grips},{-1,1})
        safe=fixture(entities[:1]);safe_sim=Simulator(safe)
        self.assertTrue(any(isinstance(row[2],ZipBoardGoal) for row in
                    _support_candidates(safe_sim,safe_sim.initial_state("a"),goal,"forward",())))

    def test_mover_opening_selection_tracks_actual_lane_and_motion_direction(self):
        from celeste_search.assist_planning import TileTargetGoal
        room=fixture().rooms["a"]
        mover=Entity("zipMover",3,8,48,24,24,{},((64.,48.),))
        state=SimState("a",64,96)
        goal=TileTargetGoal("target","a",-4,65.5,tile_x=-8,tile_y=56,
                           source_room="a",transition_to="b",edge="left")
        self.assertTrue(zip_opens_target(mover,room,state,goal))
        self.assertFalse(zip_opens_target(replace(mover,nodes=((-16.,48.),)),room,state,goal))
        self.assertFalse(zip_opens_target(mover,room,state,replace(goal,tile_y=80)))
        self.assertFalse(zip_opens_target(mover,room,state,replace(goal,transition_to="")))
        # Once the actual mover is already at its destination, no reactivation
        # or manufactured return-to-origin should be requested.
        moved=replace(state,mechanisms=(("a:3",3,20,64.,48.,0.),))
        self.assertFalse(zip_opens_target(mover,room,moved,goal))

    def test_named_direct_and_graph_routes_still_replay_exactly(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state("a")
        for strategy in ("direct_fine","direct_balanced","support_beam_forward","ordered_fine","ordered_precision"):
            with self.subTest(strategy=strategy),patch("celeste_search.assist_planning._context",return_value=(data,sim)):
                result=plan_to_route("portfolio.bin",start.to_dict(),{"waypoints":[{"room":"a","x":96,"y":88}]},
                                     seconds=2,expansions=1500,strategy=strategy,max_attempts=1)
            self.assertTrue(result["can_execute"],result)
            self.assertEqual(result["strategy"],strategy)
            state=start
            for mask in result["actions"]:
                state=sim.step(state,mask)
                self.assertFalse(state.dead)
            self.assertEqual(state.to_dict()["x"],result["final_state"]["x"])
            self.assertEqual(result["waypoint_indices"], [len(result["actions"])])

    def test_single_target_recipe_cannot_bypass_no_dash_constraints(self):
        data=fixture();sim=Simulator(data)
        with patch("celeste_search.assist_planning._context",return_value=(data,sim)):
            result=plan_to_route("portfolio.bin",sim.initial_state("a").to_dict(),
                    {"waypoints":[{"room":"a","x":96,"y":88}],"segments":[{"no_dash":True}]},
                    strategy="semantic_refill",seconds=1,max_attempts=1)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["reason"],"strategy_requires_unrestricted_single_target")
        with self.assertRaises(ValueError):
            plan_to_route("portfolio.bin",{}, {},strategy="nonexistent")

    def test_zip_goal_boards_and_rides_the_actual_dynamic_state(self):
        mover=Entity("zipMover",2,48,80,32,8,{},((96.,80.),))
        data=fixture([mover]);sim=Simulator(data);start=sim.initial_state("a")
        goal=ZipBoardGoal("zip_board","a",64,80,id="zip:a:2",identity="a:2",width=32)
        boarding=search_leg(sim,start,goal,deadline=time.monotonic()+2,max_expansions=1000,variant=4)
        self.assertTrue(boarding.reached,boarding.reason)
        ride=_ride_zip(sim,boarding.state,goal,200)
        self.assertIsNotNone(ride)
        actions,arrival=ride
        state=start
        for action in boarding.actions+actions:
            state=sim.step(state,action)
            self.assertFalse(state.dead)
        self.assertEqual(state,arrival)
        self.assertTrue(goal.achieved(arrival))
        record=next(row for row in arrival.mechanisms if row[0]=="a:2")
        self.assertEqual(record[1],3)
        self.assertEqual(round(record[3]),96)
        self.assertEqual(start.mechanisms,())

    def test_graph_exposes_zip_supports_ignored_by_static_coverage(self):
        mover=Entity("zipMover",2,48,80,32,8,{},((96.,80.),))
        data=fixture([mover]);sim=Simulator(data);start=sim.initial_state("a")
        goals=_support_candidates(sim,start,Goal("point","a",176,96),"zip_chain",())
        self.assertTrue(any(isinstance(row[2],ZipBoardGoal) for row in goals))
        without=fixture()
        self.assertFalse(any(row["family"]=="zip_graph" for row in route_strategy_specs(
                         {"waypoints":[{"room":"a","x":96,"y":88}]},start,without)))

    def test_named_zip_strategy_uses_real_relay_before_single_user_target(self):
        mover=Entity("zipMover",2,48,80,32,8,{},((96.,80.),))
        data=fixture([mover]);sim=Simulator(data)
        start=replace(sim.initial_state("a"),x=24,frame=500)
        with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
             patch.object(Simulator,"initial_state",side_effect=AssertionError("must retain actual anchor")):
            result=plan_to_route("portfolio.bin",start.to_dict(),
                    {"waypoints":[{"room":"a","x":144,"y":88}]},seconds=2,expansions=1500,
                    strategy="zip_board_ride",max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertTrue(any(stage["goal"]=="a:2" and stage["reached"]
                            for stage in result["strategy_details"]["internal_stages"]))
        state=start
        for action in result["actions"]:
            state=sim.step(state,action)
            self.assertFalse(state.dead)
        self.assertEqual(state.frame,500+len(result["actions"]))
        self.assertEqual(len(result["route"]["waypoints"]),1)
        self.assertEqual(result["waypoint_indices"],[len(result["actions"])])

    def test_exact_state_strategy_keeps_distinct_subpixel_arrivals(self):
        data=fixture();start=Simulator(data).initial_state("a")
        class SubpixelBranch:
            map=data
            def capability_report(self,name):
                return {"unsupported":[]}
            def step(self,state,action):
                if state.x==16:
                    return replace(state,x=40,rem_x=-.4 if action==1 else .4,frame=state.frame+1)
                if state.x==40:
                    return replace(state,x=80,dead=state.rem_x<0,frame=state.frame+1)
                return replace(state,dead=True,frame=state.frame+1)
        def branches(state,*args):
            return [(1,1),(2,1)] if state.x==16 else [(1,1)]
        route={"waypoints":[{"room":"a","x":40,"y":88},{"room":"a","x":80,"y":88}]}
        with patch("celeste_search.assist_planning._context",return_value=(data,SubpixelBranch())), \
             patch("celeste_search.assist_planning.macros",side_effect=branches):
            coarse=plan_to_route("portfolio.bin",start.to_dict(),route,strategy="ordered_fine",seconds=1,max_attempts=1)
            exact=plan_to_route("portfolio.bin",start.to_dict(),route,strategy="ordered_precision",seconds=1,max_attempts=1)
        self.assertFalse(coarse["can_execute"])
        self.assertTrue(exact["can_execute"],exact)
        self.assertEqual(exact["actions"],[2,1])

    def test_ordered_frontier_resumes_same_anchor_and_reanchors_absolute_frame(self):
        data=fixture();start=Simulator(data).initial_state("a")
        class TwoStep:
            map=data
            def capability_report(self,name):
                return {"unsupported":[]}
            def step(self,state,action):
                return replace(state,x=40 if state.x==16 else 80,frame=state.frame+1)
        simulator=TwoStep()
        route={"waypoints":[{"room":"a","x":40,"y":88},{"room":"a","x":80,"y":88}]}
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)), \
             patch("celeste_search.assist_planning.macros",return_value=[(1,1)]):
            first=plan_to_route("portfolio.bin",start.to_dict(),route,strategy="ordered_fine",
                                seconds=1,expansions=1,max_attempts=1)
            later=replace(start,frame=999)
            second=plan_to_route("portfolio.bin",later.to_dict(),route,strategy="ordered_fine",
                                 seconds=1,expansions=1,max_attempts=1)
            changed=plan_to_route("portfolio.bin",replace(later,rem_x=.2).to_dict(),route,
                                  strategy="ordered_fine",seconds=1,expansions=1,max_attempts=1)
        self.assertFalse(first["can_execute"])
        self.assertEqual(first["actions"],[])
        self.assertTrue(second["can_execute"],second)
        self.assertTrue(second["strategy_details"]["resumed_frontier"])
        self.assertEqual(second["actions"],[1,1])
        self.assertEqual(second["expected"][0]["frame"],999)
        self.assertEqual(second["final_state"]["frame"],1001)
        self.assertFalse(changed["strategy_details"]["resumed_frontier"])
        self.assertFalse(changed["can_execute"])

    def test_phase_frontier_preserves_and_schedules_earlier_arrivals(self):
        frontier=_OrderedFrontier(diverse=True)
        frontier.push(1000,100,0)
        for index in range(10):
            frontier.push(index,index,1)
        popped=[frontier.pop()[1] for _ in range(3)]
        self.assertIn(100,popped)
        frontier.push(1000,101,0)
        frontier.trim(2)
        self.assertEqual(len(frontier),2)
        self.assertEqual({frontier.pop()[1],frontier.pop()[1]}, {101,2})

    def test_deadline_mid_macro_retains_unexpanded_alternative_on_resume(self):
        data=fixture();start=Simulator(data).initial_state("a")
        clock=[0.0]
        class InterruptedBranch:
            map=data
            interrupted=False
            def capability_report(self,name):
                return {"unsupported":[]}
            def step(self,state,action):
                if not self.interrupted:
                    self.interrupted=True
                    clock[0]=2.0
                if state.x==16:
                    return replace(state,x=20 if action==1 else 40,frame=state.frame+1)
                return replace(state,x=80,dead=state.x==20,frame=state.frame+1)
        simulator=InterruptedBranch()
        route={"waypoints":[{"room":"a","x":40,"y":88},{"room":"a","x":80,"y":88}]}
        def choices(state,*args):
            return [(1,2),(2,1)] if state.x==16 else [(2,1)]
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)), \
             patch("celeste_search.assist_planning.macros",side_effect=choices), \
             patch("celeste_search.assist_planning.time.monotonic",side_effect=lambda:clock[0]):
            first=plan_to_route("portfolio.bin",start.to_dict(),route,strategy="ordered_fine",
                                seconds=1,expansions=4,max_attempts=1)
            second=plan_to_route("portfolio.bin",start.to_dict(),route,strategy="ordered_fine",
                                 seconds=1,expansions=4,max_attempts=1)
        self.assertFalse(first["can_execute"])
        self.assertTrue(second["can_execute"],second)
        self.assertTrue(second["strategy_details"]["resumed_frontier"])
        self.assertEqual(second["actions"],[2,2])

    def test_refill_wait_recipe_generates_real_wait_frames_before_return_target(self):
        data=fixture([Entity("refill",9,80,88,0,0)])
        start=SimState("a",16,96,dashes=0,refill_timers=(("a:9",5),))
        class Cooldown:
            map=data
            def capability_report(self,name):
                return {"unsupported":[]}
            def step(self,state,action):
                timer=dict(state.refill_timers).get("a:9",0)
                if action==0:
                    timer=max(0,timer-1)
                    return replace(state,frame=state.frame+1,dashes=0 if timer else 1,
                                   refill_timers=(("a:9",timer),) if timer else ())
                return replace(state,x=80,dead=timer>0,frame=state.frame+1)
        route={"waypoints":[{"room":"a","x":80,"y":88}]}
        with patch("celeste_search.assist_planning._context",return_value=(data,Cooldown())), \
             patch("celeste_search.assist_planning.macros",return_value=[(1,1)]):
            result=plan_to_route("portfolio.bin",start.to_dict(),route,strategy="ordered_refill_wait",
                                 seconds=1,expansions=20,max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["actions"],[0]*5+[1])
        self.assertEqual(result["final_state"]["frame"],6)

    def test_wall_return_setup_reaches_airborne_tile_using_physical_inputs(self):
        room=Room("a",0,0,240,208,tuple("1"*6+"0"*24 for _ in range(24))
                  +("1"*30,)*2,(),((64.,124.),))
        data=MapData("wall_return",{"a":room},Path("wall_return.bin"))
        simulator=Simulator(data)
        start=SimState("a",63,124,vx=-75,vy=75,dashes=0)
        goal=Goal("target","a",124,105.5)
        target={"room":"a","x":120,"y":96}
        reached=[]
        proposals=list(_wall_setup_macros(simulator,start,Heuristic(data,goal)))
        self.assertLessEqual(len(proposals),12)
        for actions,duration in proposals:
            self.assertEqual(len(actions),duration)
            self.assertLessEqual(duration,260)
            state=start
            for mask in actions:
                self.assertFalse(mask&32)
                state=simulator.step(state,mask)
                if state.dead:
                    break
                if target_reached(state,target):
                    reached.append(state)
                    break
        self.assertTrue(reached,"climb away from the target, then return through its actual tile")
        self.assertTrue(all(not state.on_ground and state.frame>start.frame for state in reached))
        self.assertEqual(start.x,63)

    def test_internal_refills_leave_the_two_explicit_waypoints_and_replay_contract_unchanged(self):
        data=fixture([Entity("refill",9,80,88,0,0)]);simulator=Simulator(data)
        start=simulator.initial_state("a")
        route=normalize_route(data,{"waypoints":[{"room":"a","x":160,"y":88},
                                                 {"room":"a","x":16,"y":88}]})
        guidance,indices=_refill_guidance_route(data,start,route)
        self.assertEqual(indices,[2,4])
        self.assertEqual([point["x"] for point in guidance["waypoints"]],[80,160,80,16])
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)):
            result=plan_to_route("portfolio.bin",start.to_dict(),route,strategy="ordered_semantic",
                                 seconds=2,expansions=1000,max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["route"],route)
        self.assertEqual(len(result["waypoint_indices"]),2)
        self.assertEqual(result["waypoint_indices"][-1],len(result["actions"]))
        self.assertEqual(result["strategy_details"]["internal_waypoints"],2)
        state=start
        for mask in result["actions"]:
            state=simulator.step(state,mask)
            self.assertFalse(state.dead)
        self.assertEqual(state,SimState.from_dict(result["final_state"]))

    def test_internal_refill_guidance_preserves_constrained_segment_history(self):
        data=fixture([Entity("refill",9,80,88,0,0)]);simulator=Simulator(data)
        route=normalize_route(data,{"waypoints":[{"room":"a","x":160,"y":88}],
                                    "segments":[{"no_ground":True}]})
        guidance,indices=_refill_guidance_route(data,simulator.initial_state("a"),route)
        self.assertIs(guidance,route)
        self.assertEqual(indices,[1])

    def test_internal_refill_milestone_requires_a_usable_dash(self):
        target={"room":"a","x":80,"y":88,"internal_refill_ready":True}
        exhausted=SimState("a",80,96,dashes=0,refill_timers=(("a:9",80),))
        self.assertFalse(target_reached(exhausted,target))
        self.assertTrue(target_reached(replace(exhausted,dashes=1),target))
        self.assertFalse(target_reached(replace(exhausted,dashes=1,x=120),target))
        # A user's ordinary selected tile still means position, irrespective of
        # dash count; this stronger requirement belongs only to internal setup.
        self.assertTrue(target_reached(exhausted,{"room":"a","x":80,"y":88}))

    def test_single_exit_search_continues_after_crossing_through_the_wrong_lane(self):
        rooms=[Room("a",0,0,64,64,("0"*8,)*8,(),((34.,7.),)),
               Room("b",0,-64,64,64,("0"*8,)*8,(),((34.,-32.),))]
        data=MapData("exit_lane",{room.name:room for room in rooms},Path("exit_lane.bin"))
        start=SimState("a",34,7)
        target=target_from_world(data,"a",34,-4)
        class TwoCrossingChoices:
            map=data
            def capability_report(self,name):
                return {"unsupported":[]}
            def step(self,state,action):
                if state.room=="a":
                    return replace(state,room="b",x=28 if action==1 else 34,y=-5,
                                   frame=state.frame+1,reason="transition")
                # Moving into the selected lane after the crossing must not
                # cause search to stop and propose a route the monitor rejects.
                return replace(state,x=34,frame=state.frame+1,reason="")
        with patch("celeste_search.assist_planning._context",return_value=(data,TwoCrossingChoices())), \
             patch("celeste_search.planner.macros",return_value=[(1,2),(2,1)]):
            result=plan_to_route("exit_lane.bin",start.to_dict(),{"waypoints":[target]},
                                 strategy="direct_greedy",seconds=1,expansions=10,max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["actions"],[2])
        self.assertEqual(result["waypoint_indices"],[1])
        self.assertEqual(result["final_state"]["x"],34)

    def test_semantic_guidance_keeps_actual_crossing_requirement_for_exits(self):
        from celeste_search.assist_routes import start_route_progress
        rooms=[Room("a",0,0,64,64,("0"*8,)*8,(),((34.,7.),)),
               Room("b",0,-64,64,64,("0"*8,)*8,(),((34.,-32.),))]
        data=MapData("exit_lane",{room.name:room for room in rooms},Path("exit_lane.bin"))
        start=SimState("a",34,7)
        target=target_from_world(data,"a",34,-4)
        captured=[]
        def inspect(simulator,state,route,goals,**options):
            captured.extend(goals)
            return [],state,False,0,0,start_route_progress(route,state),{}
        with patch("celeste_search.assist_planning._context",return_value=(data,Simulator(data))), \
             patch("celeste_search.assist_planning._search_ordered_route",side_effect=inspect):
            plan_to_route("exit_lane.bin",start.to_dict(),{"waypoints":[target]},
                          strategy="ordered_semantic",seconds=1,max_attempts=1)
        self.assertEqual(len(captured),1)
        # Walking into the right column after crossing in another column must
        # never finish a staged search; only the actual crossing can do so.
        position=replace(start,room="b",x=34,y=-5,reason="")
        self.assertFalse(captured[0].achieved(position))
        self.assertTrue(captured[0].achieved(replace(position,reason="transition")))

    def test_wall_relay_composes_physical_stages_without_resetting_the_anchor(self):
        data=fixture();simulator=Simulator(data);start=simulator.initial_state("a")
        @dataclass(frozen=True)
        class StageGoal(Goal):
            grip_offset: float=0
        # The wall module separately verifies contact/climb/landing physics.
        # This integration fixture isolates how the recipe joins two actual
        # simulated stages, retaining momentum, timers and the original anchor.
        stage=StageGoal("surface","a",48,96,id="wall:fixture")
        launches=[]
        def physical_launch(sim,state,max_frames):
            launches.append(state)
            actions=[1]*6+[0]*8
            arrival=state
            for mask in actions:
                arrival=sim.step(arrival,mask)
            return [(actions,arrival,"physical_fixture_launch")]
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)), \
             patch("celeste_search.wall_moves.wall_grip_goals",return_value=[stage]), \
             patch("celeste_search.wall_moves.wall_launch_edges",side_effect=physical_launch), \
             patch.object(Simulator,"initial_state",side_effect=AssertionError("must preserve actual state")):
            result=plan_to_route("portfolio.bin",start.to_dict(),
                                {"waypoints":[{"room":"a","x":144,"y":88}]},
                                strategy="wall_relay",seconds=2,expansions=1000,max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertTrue(launches)
        self.assertGreater(launches[0].frame,start.frame)
        self.assertEqual(SimState.from_dict(result["anchor_state"]),start)
        self.assertEqual(len(result["route"]["waypoints"]),1)
        state=start
        for mask in result["actions"]:
            state=simulator.step(state,mask)
            self.assertFalse(state.dead)
        self.assertEqual(state,SimState.from_dict(result["final_state"]))
        self.assertEqual(result["waypoint_indices"],[len(result["actions"])])

    def test_floor_and_both_wall_spring_goals_require_the_real_launch(self):
        for kind,x,y,px,vx in (("spring",48,96,48,0),
                               ("wallSpringLeft",48,64,51,-10),
                               ("wallSpringRight",96,64,93,10)):
            with self.subTest(kind=kind):
                entity=Entity(kind,9,x,y,0,0)
                data=fixture([entity]);simulator=Simulator(data)
                before=SimState("a",px,y,vx=vx,dashes=0,stamina=10)
                goal=spring_goal("a",entity)
                self.assertFalse(goal.achieved(before))
                after=simulator.step(before,0)
                self.assertTrue(goal.achieved(after),after)
                self.assertTrue(after.auto_jump_active)
                self.assertEqual(after.dashes,1)
                self.assertEqual(after.stamina,110)

    def test_spring_portfolio_filters_absent_disabled_and_restricted_cases(self):
        for attrs,enabled in (({},True),({"playerCanUse":False},False),
                              ({"playerCanUse":" FALSE "},False),
                              ({"collidable":False},False)):
            data=fixture([Entity("spring",9,48,96,0,0,attrs)])
            state=Simulator(data).initial_state("a")
            specs=route_strategy_specs({"waypoints":[{"room":"a","x":96,"y":88}]},state,data)
            self.assertEqual(any(spec["name"]=="spring_relay" for spec in specs),enabled)
        self.assertFalse(any(spec["name"]=="spring_relay" for spec in route_strategy_specs(
             {"waypoints":[{"room":"a","x":96,"y":88}]},state,fixture())))
        for route in ({"waypoints":[{"room":"a","x":96,"y":88}]*2},
                      {"waypoints":[{"room":"a","x":96,"y":88}],"segments":[{"no_dash":True}]}):
            self.assertFalse(any(spec["name"]=="spring_relay" for spec in route_strategy_specs(route,state,data)))

    def test_spring_relay_reaches_an_ordinary_air_tile_with_preserved_launch_state(self):
        data=fixture([Entity("spring",9,48,96,0,0)]);simulator=Simulator(data)
        start=simulator.initial_state("a")
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)), \
             patch.object(Simulator,"initial_state",side_effect=AssertionError("no reset")):
            result=plan_to_route("portfolio.bin",start.to_dict(),
                                {"waypoints":[{"room":"a","x":104,"y":32}]},
                                strategy="spring_relay",seconds=2,expansions=1500,max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertTrue(result["strategy_details"]["spring_launch"])
        self.assertTrue(any(stage["actual_launch"] for stage in result["strategy_details"]["internal_stages"]))
        self.assertEqual(len(result["route"]["waypoints"]),1)
        state=start;launched=False
        for mask in result["actions"]:
            state=simulator.step(state,mask)
            self.assertFalse(state.dead)
            launched=launched or spring_goal("a",data.rooms["a"].entities[0]).achieved(state)
        self.assertTrue(launched)
        self.assertEqual(state,SimState.from_dict(result["final_state"]))

    def test_spring_launch_without_a_complete_suffix_never_executes(self):
        from celeste_search.planner import SearchResult
        data=fixture([Entity("spring",9,48,96,0,0)]);simulator=Simulator(data)
        start=simulator.initial_state("a")
        def incomplete(sim,state,goal,**kwargs):
            return SearchResult([],state,False,0,0,"budget")
        def approach_only(sim,state,goal,**kwargs):
            return search_leg(sim,state,goal,**kwargs) if goal.kind=="spring" else incomplete(sim,state,goal)
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)), \
             patch("celeste_search.assist_strategies.search_leg",side_effect=approach_only), \
             patch("celeste_search.strategies.search_goal",side_effect=incomplete):
            result=plan_to_route("portfolio.bin",start.to_dict(),
                                {"waypoints":[{"room":"a","x":104,"y":32}]},
                                strategy="spring_relay",seconds=1,expansions=1000,max_attempts=1)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["actions"],[])
        self.assertTrue(result["strategy_details"]["spring_launch"])

    def test_transient_support_uses_live_position_and_rejects_absent_phases(self):
        for kind,absent_phase in (("crumbleBlock",3),("fallingBlock",4)):
            goal=TransientSupportGoal("surface","a",64,80,id="support:a:8",
                    identity="a:8",width=32,mechanism_kind=kind)
            state=SimState("a",80,72,on_ground=True,
                           mechanisms=(("a:8",1,9,64.,72.,0.),))
            self.assertTrue(goal.achieved(state))
            self.assertEqual(goal.waypoint(state),(80,72))
            self.assertFalse(goal.achieved(replace(state,vy=-10)))
            self.assertFalse(goal.achieved(replace(state,y=80)))
            self.assertFalse(goal.achieved(replace(state,
                      mechanisms=(("a:8",absent_phase,9,64.,72.,0.),))))
            self.assertFalse(goal.achieved(replace(state,broken=frozenset({"a:8"}))))

    def test_coast_landing_retains_real_temporary_support_timers(self):
        # Start airborne above a temporary support: the branch must actually
        # land and retain its ticking mechanism instead of resetting to a
        # generic surface position or a fully rested state.
        data=fixture([Entity("crumbleBlock",8,48,72,48,8)])
        simulator=Simulator(data)
        start=SimState("a",72,48,vy=60,dashes=0,stamina=40,frame=123)
        stats=dict(expansions=0,simulated_frames=0)
        arrivals=list(_coast_support_arrivals(simulator,start,Goal("target","a",160,88),
                         deadline=time.monotonic()+1,max_frames=40,stats=stats))
        on_crumble=[row for row in arrivals if row[1].y==72]
        self.assertTrue(on_crumble)
        for actions,arrival,_ in on_crumble:
            replay=start
            for mask in actions:
                replay=simulator.step(replay,mask)
            self.assertEqual(replay,arrival)
            self.assertEqual(arrival.frame,start.frame+len(actions))
            self.assertTrue(any(record[0]=="a:8" for record in arrival.mechanisms))
            self.assertFalse(arrival.dead)
        self.assertGreater(stats["simulated_frames"],0)
        self.assertLessEqual(stats["simulated_frames"],120)

    def test_underhang_portfolio_preserves_restricted_route_rules(self):
        point={"room":"a","x":96,"y":88}
        self.assertTrue(any(spec["name"]=="underhang_relay"
                            for spec in route_strategy_specs({"waypoints":[point]})))
        for route in ({"waypoints":[point,point]},
                      {"waypoints":[point],"segments":[{"no_dash":True}]},
                      {"waypoints":[point],"segments":[{"no_ground":True}]}):
            self.assertFalse(any(spec["name"]=="underhang_relay"
                                 for spec in route_strategy_specs(route)))
        data=fixture();simulator=Simulator(data)
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)):
            result=plan_to_route("portfolio.bin",simulator.initial_state("a").to_dict(),
                      {"waypoints":[point],"segments":[{"no_dash":True}]},
                      strategy="underhang_relay",seconds=1,max_attempts=1)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["reason"],"strategy_requires_unrestricted_single_target")

    def test_underhang_relay_physically_uses_spring_and_crumble_then_exact_tile(self):
        from celeste_search.search_progress import progress_reporter
        data=underhang_fixture();simulator=Simulator(data);start=simulator.initial_state("a")
        route={"waypoints":[{"room":"a","x":112,"y":160}]}
        packets=[]
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)), \
             patch.object(Simulator,"initial_state",side_effect=AssertionError("must use actual anchor")), \
             progress_reporter(packets.append,start,normalize_route(data,route),interval=0):
            result=plan_to_route("underhang.bin",start.to_dict(),route,strategy="underhang_relay",
                                seconds=1,expansions=100,max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(len(result["route"]["waypoints"]),1)
        self.assertEqual(result["waypoint_indices"],[len(result["actions"])])
        replay=start;launched=landed=False
        spring=spring_goal("a",data.rooms["a"].entities[3])
        for mask in result["actions"]:
            replay=simulator.step(replay,mask)
            self.assertFalse(replay.dead)
            launched=launched or spring.achieved(replay)
            landed=landed or (replay.on_ground and replay.vy>=0 and replay.y==120
                              and any(row[0]=="a:30" and row[1]!=3 for row in replay.mechanisms))
        self.assertTrue(launched)
        self.assertTrue(landed)
        self.assertEqual(replay,SimState.from_dict(result["final_state"]))
        self.assertEqual(SimState.from_dict(result["anchor_state"]),start)
        self.assertTrue(target_reached(replay,result["route"]["waypoints"][0]))
        self.assertTrue(packets)
        for packet in packets:
            for path in packet["paths"]:
                self.assertFalse(path["local_leg"])
                self.assertEqual(path["points"][0],{"room":"a","x":start.x,"y":start.y})
                self.assertLessEqual(len(path["points"]),120)

    def test_underhang_approach_without_full_target_hit_cannot_execute(self):
        data=underhang_fixture();simulator=Simulator(data);start=simulator.initial_state("a")
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)):
            result=plan_to_route("underhang.bin",start.to_dict(),
                    {"waypoints":[{"room":"a","x":112,"y":160}]},strategy="underhang_relay",
                    seconds=.3,expansions=80,max_attempts=1,max_frames=40)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["actions"],[])


if __name__=="__main__":
    unittest.main()
