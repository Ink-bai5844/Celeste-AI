"""Real, resumable dash collisions with a specific destructible passage.

The geometry planner supplies an approach on the occupied side of a block.
Every candidate below keeps the actual collision rebound and resource state;
neither a nearby coordinate nor another broken block satisfies this edge.
"""
from dataclasses import replace

from .macro_continuation import MACRO_EXPANSION, MacroContinuation
from .physics import DASH, DOWN, JUMP, LEFT, RIGHT, UP, _truth
from .search_progress import remember_segment


def _recipes(origin, dx, dy):
    horizontal=RIGHT if dx>0 else LEFT if dx<0 else 0
    vertical=DOWN if dy>0 else UP if dy<0 else 0
    aim=horizontal|vertical
    if origin.dash_attack:
        yield (aim,)*36,"existing_attack"
    if origin.dashes<=0:
        return
    aims=(aim, aim|UP, aim|DOWN) if horizontal and not vertical else (
        (aim, aim|LEFT, aim|RIGHT) if vertical and not horizontal else (aim,horizontal,vertical))
    release=int(bool(origin.prev_input&DASH))
    for wait in (0,2,6,12,18):
        for direction in dict.fromkeys(aims):
            if direction:
                yield ((0,)*max(release,wait)+(DASH|direction,)*6+(direction,)*30,
                       f"dash:{direction}:wait{wait}")
    for wait in (0,4,8,12):
        for direction in dict.fromkeys(aims):
            if direction:
                yield ((0,)*release+(JUMP|horizontal,)*wait+
                       (DASH|direction,)*6+(direction,)*30,
                       f"jump_dash:{direction}:jump{wait}")


def dash_block_edges(simulator, origin, target, *, deadline, max_frames=160,
                     stats=None, max_expansions=None, max_edges=4,
                     continuation=None, remember_results=True):
    """Return exact inputs and post-collision states for the named block.

    An optional caller-owned continuation preserves both recipe enumeration
    and a partially simulated recipe. A time slice never changes the origin
    or grants a dash. No-dash callers may only reuse an already broken block.
    An explicit public-goal wrapper can end the edge before the collision;
    that arrival does not claim that the block was broken.
    """
    own=continuation is None
    handle=continuation or MacroContinuation()
    if stats is None:
        stats=handle.__dict__.setdefault("stats",{})
    key=(id(simulator),id(stats),origin,target,max_frames,max_edges,remember_results)

    def work(current_handle):
        identity=getattr(target,"identity","")
        data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
        room=data.rooms.get(origin.room)
        entity=next((entity for entity in room.entities
                     if f"{room.name}:{entity.id}"==identity),None) if room else None
        if (origin.dead or max_frames<=0 or max_edges<=0 or entity is None
                or entity.kind!="dashBlock"):
            return []
        if identity in origin.broken:
            return [([],origin,"dash_block:already_broken")]
        phase=getattr(simulator,"phase",None)
        segments=getattr(simulator,"route",{}).get("segments",())
        segment_no_dash=bool(phase is not None and phase.index<len(segments)
                             and segments[phase.index].get("no_dash"))
        if (not _truth(entity.attrs.get("canDash",True))
                or getattr(simulator,"no_dash",False) or segment_no_dash):
            return []
        dx,dy=getattr(target,"dash_dx",0),getattr(target,"dash_dy",0)
        if dx not in (-1,0,1) or dy not in (-1,0,1) or not (dx or dy):
            return []
        seen, recipes_seen, published = {},set(),set()
        public=getattr(target,"public_goal",None)

        def results():
            rows=sorted(seen.values(),key=lambda row:row[0])[:max_edges]
            for _,actions,state,_,states in rows:
                action_key=tuple(actions)
                if remember_results and action_key not in published:
                    remember_segment(origin,state,states)
                    published.add(action_key)
            return [(actions,state,label) for _,actions,state,label,_ in rows]

        for recipe,label in _recipes(origin,dx,dy):
            recipe=recipe[:max_frames]
            if recipe in recipes_seen:
                continue
            recipes_seen.add(recipe)
            yield MACRO_EXPANSION
            state,actions,states,dashes=origin,[],[origin],[]
            for tick,mask in enumerate(recipe):
                if tick%16==0:
                    yield None
                state=simulator.step(state,mask)
                started_dash=bool(getattr(simulator,"last_step_started_dash",False))
                if started_dash:
                    dashes.append(tick)
                stats["simulated_frames"]=stats.get("simulated_frames",0)+1
                actions.append(mask)
                states.append(state)
                if state.dead:
                    break
                public_hit=(public is not None and public.achieved(state)
                    and not getattr(public,"first_hit_rejected",lambda state:False)(state))
                if identity in state.broken or public_hit:
                    # Only the simulator can add this identity. An earlier
                    # public hit preserves its actual unbroken state instead.
                    # Keep rebound velocity, timers, expenditure and clocks.
                    fingerprint=replace(state,frame=0,reason="")
                    rank=(len(dashes),-state.dashes,-state.stamina,len(actions),
                          tuple(-frame for frame in dashes))
                    if fingerprint not in seen or rank<seen[fingerprint][0]:
                        event="public_hit:" if public_hit and identity not in state.broken else ""
                        seen[fingerprint]=(rank,actions,state,"dash_block:"+event+label,states)
                        current_handle.results=results()
                    break
                if state.room!=origin.room:
                    break
        return results()

    try:
        return handle.advance(key,work,deadline=deadline,stats=stats,
            max_expansions=max_expansions,recipe_limit=28)
    finally:
        if own:
            handle.release()
