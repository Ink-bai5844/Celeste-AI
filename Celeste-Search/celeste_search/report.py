"""Self-contained, offline HTML inspection of a predicted Celeste route."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from .maps import MapData, TILE_SIZE


def _number(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _fmt(value):
    return f"{_number(value):.3f}".rstrip("0").rstrip(".") or "0"


def _json(value):
    # A script element terminates on </script> even inside a JSON string.
    return json.dumps(value, ensure_ascii=False, default=str).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _solid_path(room):
    paths = []
    for ty, row in enumerate(room.solids):
        tx = 0
        while tx < len(row):
            if not room.tile_at(tx, ty):
                tx += 1
                continue
            start = tx
            while tx < len(row) and room.tile_at(tx, ty):
                tx += 1
            x, y = room.x + start * TILE_SIZE, room.y + ty * TILE_SIZE
            paths.append(f"M{x} {y}h{(tx-start)*TILE_SIZE}v8h{-((tx-start)*TILE_SIZE)}z")
    return "".join(paths)


def _room_svg(room, berry_ids, visited, traces):
    name = escape(room.name, quote=True)
    group = [f'<g class="room{ " visited" if room.name in visited else ""}{ " dummy" if room.dummy else ""}" data-room="{name}">']
    group.append(f'<rect class="room-bounds" x="{room.x}" y="{room.y}" width="{room.width}" height="{room.height}"/>')
    group.append(f'<path class="solid" d="{_solid_path(room)}"/>')
    for entity in room.hazard_entities:
        if entity.kind.startswith("spikes"):
            x, y, w, h = entity.x, entity.y, entity.width or 4, entity.height or 4
            if entity.kind == "spikesUp":
                y -= 4
            if entity.kind == "spikesLeft":
                x -= 4
            group.append(f'<rect class="hazard" x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}"/>')
    for berry in room.berries:
        identity = berry.berry_id(room.name)
        predicted = identity in berry_ids
        category = berry.berry_category or "unknown"
        css = "berry predicted" if predicted else "berry pending"
        if category == "golden":
            css += " golden"
        title = escape(f"{identity} · {category} · {'模拟预测已收集' if predicted else '待收集 / 未预测'}")
        group.append(f'<g class="{css}" transform="translate({_fmt(berry.x)} {_fmt(berry.y)})"><title>{title}</title><circle r="5"/><path d="M-3 -5L0 -8L3 -5"/></g>')
        for x, y in berry.nodes:
            group.append(f'<circle class="seed" cx="{_fmt(x)}" cy="{_fmt(y)}" r="2"/>')
    for segment in traces.get(room.name, []):
        if not segment:
            continue
        path = "M" + "L".join(f'{_fmt(s.get("x"))} {_fmt(s.get("y"))}' for s in segment)
        group.append(f'<path class="route-path" d="{path}"/>')
    for x, y in room.spawns:
        group.append(f'<circle class="spawn" cx="{_fmt(x)}" cy="{_fmt(y)}" r="3"/>')
    group.append(f'<text class="room-label" x="{room.x+8}" y="{room.y+15}">{escape(room.name)}</text></g>')
    return "".join(group)


def write_report(map_data: MapData, route: dict, output: Path) -> Path:
    """Write a local report. All progress is explicitly simulator prediction.

    This function neither opens a browser nor contacts a remote service.
    Route/map labels are escaped, and embedded JSON cannot close its script.
    """
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    predicted = route.get("predicted") or {}
    berry_ids = set(map(str, predicted.get("berries") or []))
    visited = set(map(str, predicted.get("rooms") or []))
    coverage = predicted.get("coverage_cells") or []
    search = route.get("search") or {}
    capabilities = route.get("capabilities") or {}
    inventory = map_data.to_dict()
    total_berries = inventory["strawberry_count"]
    total_rooms = inventory["playable_room_count"]
    known_berries = {e.berry_id(r.name) for r in map_data.rooms.values() for e in r.berries}
    playable_rooms = {r.name for r in map_data.rooms.values() if not r.dummy}
    categories = inventory["strawberry_categories"]
    traces = {}
    last_room = None
    samples = route.get("trace") or []
    # Keep discontinuities between rooms; never draw a straight teleport line.
    for sample in samples:
        room_name = str(sample.get("room", ""))
        if room_name not in map_data.rooms:
            continue
        if room_name != last_room:
            traces.setdefault(room_name, []).append([])
        traces[room_name][-1].append(sample)
        last_room = room_name
    rooms_svg = "".join(_room_svg(r, berry_ids, visited, traces) for r in map_data.rooms.values())
    min_x = min(r.x for r in map_data.rooms.values())
    min_y = min(r.y for r in map_data.rooms.values())
    max_x = max(r.x + r.width for r in map_data.rooms.values())
    max_y = max(r.y + r.height for r in map_data.rooms.values())
    overall = [min_x-24, min_y-24, max_x-min_x+48, max_y-min_y+48]
    room_options = ['<option value="">整章地图</option>']
    for room in map_data.rooms.values():
        label = f'{room.name} · {"预测到达" if room.name in visited else "未预测到达"} · {len(room.berries)} 颗候选'
        if room.dummy:
            label += " · 装饰房间"
        room_options.append(f'<option value="{escape(room.name, quote=True)}">{escape(label)}</option>')
    legs = search.get("legs") or []
    leg_rows = []
    for index, leg in enumerate(legs):
        reached = bool(leg.get("reached", False))
        leg_rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
            index+1, leg.get("goal", ""), leg.get("kind", ""),
            "模拟达到" if reached else "未达到", leg.get("reason", ""),
        )) + "</tr>")
    if not leg_rows:
        leg_rows = ['<tr><td colspan="5" class="muted">暂无搜索阶段记录</td></tr>']
    sources = []
    for source in route.get("sources") or []:
        source = str(source)
        if urlparse(source).scheme in {"http", "https"}:
            sources.append(f'<li><a href="{escape(source, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(source)}</a></li>')
        else:
            sources.append(f"<li>{escape(source)}</li>")
    source_html = "".join(sources) or '<li class="muted">路线文件未附来源列表。</li>'
    unsupported = capabilities.get("unsupported_mechanics", capabilities.get("unsupported", inventory["unsupported_mechanics"]))
    pending = search.get("pending_goals", [])
    skipped = search.get("skipped", [])
    detail_json = escape(json.dumps({"capabilities": capabilities, "unsupported_mechanics": unsupported,
                                   "pending_goals": pending, "skipped": skipped}, ensure_ascii=False, indent=2, default=str))
    title = escape(map_data.package)
    category_text = " / ".join(f"{escape(str(k))}: {v}" for k, v in categories.items()) or "无"
    frames = int(_number(route.get("frames", 0)))
    expansions = int(_number(search.get("expansions", 0)))
    sim_frames = int(_number(search.get("simulated_frames", 0)))
    wall = _number(search.get("wall_seconds", 0))
    start_room = str(route.get("start_room") or map_data.start_room().name)
    metadata = _json({"rooms": {r.name: list(r.bounds) for r in map_data.rooms.values()},
                      "overall": overall, "start": start_room})
    document = _TEMPLATE.replace("@@TITLE@@", title).replace("@@ROOM_OPTIONS@@", "".join(room_options))
    values = {
        "@@BERRY_PROGRESS@@": f"{len(berry_ids & known_berries)} / {total_berries}",
        "@@ROOM_PROGRESS@@": f"{len(visited & playable_rooms)} / {total_rooms}",
        "@@CELLS@@": str(len(coverage)), "@@FRAMES@@": f"{frames:,}",
        "@@SECONDS@@": f"{frames/60:.1f}", "@@EXPANSIONS@@": f"{expansions:,}",
        "@@SIM_FRAMES@@": f"{sim_frames:,}", "@@WALL@@": f"{wall:.1f}",
        "@@CATEGORIES@@": category_text, "@@MAP_SVG@@": rooms_svg,
        "@@VIEWBOX@@": " ".join(map(str, overall)), "@@LEGS@@": "".join(leg_rows),
        "@@DETAILS@@": detail_json, "@@SOURCES@@": source_html, "@@METADATA@@": metadata,
        "@@TRACE_COUNT@@": f"{len(samples):,}",
        "@@START_SCOPE@@": escape(("章节起点连续路线" if start_room==map_data.start_room().name else "单房间练习路线")+f" · 起始房间 {start_room}"),
        "@@GENERATED@@": escape(datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M UTC+8")),
    }
    for key, value in values.items():
        document = document.replace(key, value)
    output.write_text(document, encoding="utf-8")
    return output


_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>Celeste 离线路线 · @@TITLE@@</title>
<style>
:root{--bg:#0d1321;--panel:#151e30;--line:#29364e;--text:#edf3fc;--muted:#9aaecb;--mint:#6ce6be;--rose:#ff718b;--gold:#ffd178;--blue:#80b6ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 15% 0%,#1e3150 0,transparent 50%),var(--bg);color:var(--text);font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:32px 28px 48px}.eyebrow{font-size:12px;letter-spacing:.14em;color:var(--mint);font-weight:700}h1{font-size:30px;line-height:1.2;margin:8px 0 10px}h2{font-size:17px;margin:0 0 12px}p{margin:8px 0}.muted{color:var(--muted)}.intro{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.badge{background:#513d1b;border:1px solid #8b6933;color:#ffe0a1;border-radius:8px;padding:9px 14px;white-space:nowrap;font-weight:700}.notice{margin:20px 0;padding:13px 16px;border-left:3px solid var(--gold);background:#2b261f;color:#eedac2;border-radius:0 7px 7px 0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.stat{background:var(--panel);border:1px solid var(--line);padding:18px 20px;border-radius:11px}.stat .label{font-size:12px;color:var(--muted)}.stat .value{font-size:27px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.45}.stat small{color:var(--muted)}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin-top:18px;padding:18px}.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px}select,button{font:inherit;color:var(--text);background:#202d43;border:1px solid #394964;border-radius:7px;padding:7px 11px}select{min-width:250px;max-width:100%}button{cursor:pointer}button:hover,select:hover{border-color:var(--blue)}button:focus-visible,select:focus-visible{outline:2px solid var(--mint);outline-offset:2px}label{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--muted);margin-left:auto}.viewport{height:560px;min-height:360px;border:1px solid #293a54;border-radius:8px;overflow:hidden;background:#0a111d;position:relative}.viewport svg{width:100%;height:100%;touch-action:none;cursor:grab}.viewport svg.dragging{cursor:grabbing}.room-bounds{fill:#111e31;stroke:#405678;stroke-width:1}.room.visited .room-bounds{fill:#102832;stroke:#386b67}.room.dummy{opacity:.35}.solid{fill:#354864;stroke:none}.hazard{fill:#b94761;opacity:.9}.berry circle{fill:var(--rose);stroke:#ffd1da;stroke-width:1}.berry path{fill:none;stroke:#80e1a6;stroke-width:2}.berry.golden circle{fill:var(--gold);stroke:#fff0bc}.berry.predicted circle{fill:var(--mint);stroke:#d8fff0}.seed{fill:#a6d7ff}.route-path{fill:none;stroke:#76ead3;stroke-width:1.5;stroke-linejoin:round;stroke-linecap:round;opacity:.9;pointer-events:none}.spawn{fill:none;stroke:#b7d9ff;stroke-width:1}.room-label{fill:#c4d6ee;font-size:10px;font-family:system-ui;paint-order:stroke;stroke:#111e31;stroke-width:2;pointer-events:none}.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:var(--muted)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.coordinates{font-size:12px;font-variant-numeric:tabular-nums;margin-top:8px;color:#c4d6ee;min-height:20px}.caption{font-size:12px;color:var(--muted)}.grid2{display:grid;grid-template-columns:1.3fr 1fr;gap:18px}.grid2 .panel{min-width:0}table{width:100%;border-collapse:collapse;text-align:left;font-size:13px}th{font-weight:600;color:var(--muted)}td,th{padding:8px;border-bottom:1px solid var(--line);vertical-align:top}td{word-break:break-word}.scroll{max-height:360px;overflow:auto}details summary{cursor:pointer;color:#bdd5f8}pre{font:12px/1.55 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#101828;border-radius:8px;padding:14px;max-height:400px;overflow:auto}a{color:#9dc7ff;overflow-wrap:anywhere}ul{padding-left:20px}footer{margin-top:22px;color:var(--muted);font-size:12px}.hidden-route .route-path{display:none}
@media(max-width:800px){main{padding:20px 14px}.intro{display:block}.badge{display:inline-block;margin-top:8px}.stats{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr;gap:0}.viewport{height:440px}h1{font-size:25px}label{margin-left:0}}
</style></head><body><main>
<header class="intro"><div><div class="eyebrow">CELESTE / OFFLINE ROUTE SEARCH</div><h1>@@TITLE@@</h1><p class="muted">@@START_SCOPE@@ · 本地物理模拟与输入序列搜索</p></div><div class="badge">仅模拟预测 · 尚未实机验证</div></header>
<div class="notice"><strong>这些结果不是游戏内的通关或收集记录。</strong> 路线由离线近似物理计算，地图总数来自静态解析；特殊机关、过场和未支持的机制可能使实际回放不同。预测命中草莓不等于已写入真实存档。</div>
<section class="stats" aria-label="模拟预测统计"><div class="stat"><div class="label">草莓预测 / 静态候选</div><div class="value">@@BERRY_PROGRESS@@</div><small>包含特殊类型，详见下方清单</small></div><div class="stat"><div class="label">预测到达 / 可进入房间清单</div><div class="value">@@ROOM_PROGRESS@@</div><small>未知机制仍可能阻挡通行</small></div><div class="stat"><div class="label">经过的 16px 网格</div><div class="value">@@CELLS@@</div><small>探索样本数，不表示覆盖率</small></div><div class="stat"><div class="label">候选输入帧数</div><div class="value">@@FRAMES@@</div><small>60 FPS 下约 @@SECONDS@@ 秒</small></div></section>
<section class="panel"><h2>地图与预测轨迹</h2><div class="toolbar"><select id="room-select" aria-label="选择房间">@@ROOM_OPTIONS@@</select><button id="overview">整章</button><button id="reset">适合视图</button><button id="zoom-in" aria-label="放大">＋</button><button id="zoom-out" aria-label="缩小">−</button><label><input id="show-route" type="checkbox" checked>显示预测轨迹</label></div>
<div class="viewport"><svg id="map" xmlns="http://www.w3.org/2000/svg" viewBox="@@VIEWBOX@@" role="img" aria-label="静态地图、草莓与离线预测路径">@@MAP_SVG@@</svg></div>
<div class="legend"><span><i class="dot" style="background:var(--mint)"></i>预测收集 / 路径</span><span><i class="dot" style="background:var(--rose)"></i>待收集草莓</span><span><i class="dot" style="background:var(--gold)"></i>金莓候选</span><span><i class="dot" style="background:#b94761"></i>静态尖刺</span><span>○ 出生点</span></div>
<div id="coordinates" class="coordinates">拖动平移，滚轮缩放；悬停查看世界坐标。</div><p class="caption">地形按原生 8px 瓦片显示。轨迹含 @@TRACE_COUNT@@ 个采样点；折线连接采样点，不代表帧间精确碰撞路径。总量分类：@@CATEGORIES@@。</p></section>
<div class="grid2"><section class="panel"><h2>搜索阶段</h2><p class="caption">扩展 @@EXPANSIONS@@ 次 · 离线推进 @@SIM_FRAMES@@ 帧 · 耗时 @@WALL@@ 秒</p><div class="scroll"><table><thead><tr><th>#</th><th>目标</th><th>类型</th><th>预测结果</th><th>说明</th></tr></thead><tbody>@@LEGS@@</tbody></table></div></section>
<section class="panel"><h2>支持范围与待办</h2><p class="muted">未实现或保守处理的机关会限制路线可靠性。请结合能力报告查看预测边界。</p><details open><summary>能力、未完成目标与跳过项</summary><pre>@@DETAILS@@</pre></details></section></div>
<section class="panel"><details><summary>源码与格式参考</summary><ul>@@SOURCES@@</ul></details></section><footer>生成于 @@GENERATED@@ · 所有图形与交互均包含在本文件中，可离线打开。链接仅在点击时访问。</footer>
</main><script id="map-data" type="application/json">@@METADATA@@</script><script>
(()=>{'use strict';const data=JSON.parse(document.getElementById('map-data').textContent),svg=document.getElementById('map'),select=document.getElementById('room-select'),coordinates=document.getElementById('coordinates');let view=data.overall.slice(),drag=null;
const setView=v=>{view=v;svg.setAttribute('viewBox',v.join(' '));};const fit=()=>{const room=data.rooms[select.value];setView(room?[room[0]-16,room[1]-16,room[2]+32,room[3]+32]:data.overall.slice());};
const change=()=>{for(const group of svg.querySelectorAll('[data-room]'))group.style.display=!select.value||group.dataset.room===select.value?'':'none';fit();};
const point=event=>{const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;const matrix=svg.getScreenCTM();return matrix?p.matrixTransform(matrix.inverse()):{x:0,y:0};};
const zoom=(factor,center)=>{const c=center||{x:view[0]+view[2]/2,y:view[1]+view[3]/2};const width=Math.min(200000,Math.max(24,view[2]*factor)),ratio=width/view[2];setView([c.x+(view[0]-c.x)*ratio,c.y+(view[1]-c.y)*ratio,width,view[3]*ratio]);};
select.addEventListener('change',change);document.getElementById('overview').onclick=()=>{select.value='';change();};document.getElementById('reset').onclick=fit;document.getElementById('zoom-in').onclick=()=>zoom(.75);document.getElementById('zoom-out').onclick=()=>zoom(1.333333);document.getElementById('show-route').onchange=e=>svg.classList.toggle('hidden-route',!e.target.checked);
svg.addEventListener('wheel',e=>{e.preventDefault();zoom(e.deltaY>0?1.12:.892857,point(e));},{passive:false});svg.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY,view:view.slice()};svg.setPointerCapture(e.pointerId);svg.classList.add('dragging');});
svg.addEventListener('pointermove',e=>{if(drag){const rect=svg.getBoundingClientRect(),scale=Math.max(drag.view[2]/rect.width,drag.view[3]/rect.height);setView([drag.view[0]-(e.clientX-drag.x)*scale,drag.view[1]-(e.clientY-drag.y)*scale,drag.view[2],drag.view[3]]);}const p=point(e);let name='房间外';for(const [key,b]of Object.entries(data.rooms)){if(p.x>=b[0]&&p.y>=b[1]&&p.x<b[0]+b[2]&&p.y<b[1]+b[3]){name=key;break;}}coordinates.textContent='房间 '+name+' · 世界坐标 ('+p.x.toFixed(1)+', '+p.y.toFixed(1)+')';});
const stop=()=>{drag=null;svg.classList.remove('dragging');};svg.addEventListener('pointerup',stop);svg.addEventListener('pointercancel',stop);if(Object.prototype.hasOwnProperty.call(data.rooms,data.start))select.value=data.start;change();})();
</script></body></html>'''
