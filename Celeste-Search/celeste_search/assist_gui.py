"""Small selectable map window for Celeste's explicitly requested assistance."""
from __future__ import annotations

import json
import colorsys
import math
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import xml.etree.ElementTree as ET

from .assist import SharedAssistState, start_assist_service
from .assist_reconnect import ReconnectRequest
from .maps import load_map
from .playback import configure_game, install_bridge, restore_configuration
from .game_platform import launch_game_process
from .assist_validation import target_from_world,AssistValidationError
from .assist_map_geometry import (MOVING_MAP_KINDS, map_hazard_rect,
                                  moving_map_geometry, moving_spike_owners,
                                  falling_map_geometry, falling_spike_owners)
from .assist_targets import room_exit_targets,expand_exit_target
from .search_progress import MAX_CORRIDOR_PREVIEWS,dash_break_conditions


CONTROL_HINT="辅助不会暂停游戏；按游戏操作键会取消接管。F12 可断开，点击右上角「重新连接」恢复连接。"
DRAFT_HINT="限制随终点移动，调整顺序后请核对；编辑草稿不会改变正在执行的路线。"
PREVIEW_INTERVAL=.4
FALLING_REFRESH_MS=33
MAP_OBSERVATION_MAX_AGE=2.0
DASH_CONDITION_COLOR="#d4a65c"


def place_preview_label(x,y,width,height,occupied,bounds):
    """Try a fixed set of nearby screen positions; leave the path endpoint exact."""
    left,top,right,bottom=bounds
    candidates=[]
    for column in range(2):
        for row in range(3):
            dx=6+column*(width+8)
            dy=5+row*(height+4)
            for px,py in ((x+dx,y-height-dy),(x+dx,y+dy),(x-width-dx,y-height-dy),(x-width-dx,y+dy)):
                px=max(left,min(px,right-width))
                py=max(top,min(py,bottom-height))
                box=(px,py,px+width,py+height)
                overlap=sum(max(0,min(box[2]+2,b[2])-max(box[0]-2,b[0]))*
                            max(0,min(box[3]+2,b[3])-max(box[1]-2,b[1])) for b in occupied)
                distance=(px+width/2-x)**2+(py+height/2-y)**2
                candidates.append((overlap,distance,box))
    box=min(candidates,key=lambda item:item[:2])[2]
    occupied.append(box)
    return box[0]+2,box[1]+1


def preview_worker_key(entry):
    return str(entry.get("worker_id",entry.get("pid",entry.get("strategy","unknown"))))


class PreviewStyles:
    """Keep one short label and color per worker for the window's lifetime."""
    PALETTE=("#fb923c","#a78bfa","#4ade80","#f472b6","#facc15","#60a5fa",
             "#2dd4bf","#e879f9","#a3e635","#fda4af","#c4b5fd","#fcd34d")

    def __init__(self):
        self.values={}

    def get(self,worker):
        worker=str(worker)
        if worker not in self.values:
            index=len(self.values)
            if index<len(self.PALETTE):
                color=self.PALETTE[index]
            else:
                rgb=colorsys.hsv_to_rgb((index*.61803398875)%1,.5,.95)
                color="#"+"".join(f"{round(channel*255):02x}" for channel in rgb)
            self.values[worker]=(f"W{index+1:02}",color)
        return self.values[worker]


def preview_room_segments(points,room,max_points=180):
    """Decimate a displayed path without connecting separate room visits."""
    chunks=[]
    current=[]
    for point in points:
        valid=(isinstance(point,dict) and point.get("room")==room
               and all(type(point.get(key)) in (int,float) and math.isfinite(point[key]) for key in ("x","y")))
        if valid:
            current.append((point["x"],point["y"]-5))
        elif current:
            chunks.append(current)
            current=[]
    if current:
        chunks.append(current)
    if not chunks:
        return []
    def sample(values,count):
        if len(values)<=count:
            return values
        if count<=1:
            return [values[-1]]
        return [values[round(index*(len(values)-1)/(count-1))] for index in range(count)]
    max_points=max(2,int(max_points))
    chunks=sample(chunks,max(1,max_points//2))
    allowance=max(2,max_points//len(chunks))
    return [sample(chunk,allowance) for chunk in chunks]


def current_search_previews(state):
    """Partial paths are a visual layer only and disappear outside search."""
    if state.get("mode") not in {"waiting","planning"}:
        return []
    entries=state.get("search_previews") or []
    if not isinstance(entries,list):
        return []
    generation=state.get("search_generation",(state.get("search_stats") or {}).get("generation"))
    return [entry for entry in entries if isinstance(entry,dict)
            and (isinstance(entry.get("paths"),list) and any(isinstance(path,dict)
                    and isinstance(path.get("points"),list) and path["points"] for path in entry["paths"])
                 or preview_corridors(entry))
            and (generation is None or entry.get("generation")==generation)]


def preview_corridors(entry):
    """Geometry is a separate visual inventory, never a simulated path."""
    topology=entry.get("topology")
    if not isinstance(topology,dict) or not isinstance(topology.get("corridors"),list):
        return []
    return [item for item in topology["corridors"][:MAX_CORRIDOR_PREVIEWS] if isinstance(item,dict)
            and isinstance(item.get("id"),str) and isinstance(item.get("points"),list)
            and item.get("status") in {"pending","trying","failed","verified"}]


def corridor_caption(corridor,index,total,cleared_dash_blocks=(),unbreakable_dash_blocks=()):
    turns=corridor.get("turns",0)
    turns=turns if type(turns) is int and turns>=0 else 0
    status={"pending":"等待下一轮" if turns else "尚未尝试",
            "trying":"验证中","failed":"本次未通过","verified":"整路已验证"}
    text=f"通路 {index}/{total} · {status[corridor['status']]}"
    conditions=dash_break_conditions(corridor)
    if conditions:
        pending=[item for item in conditions if item["identity"] not in cleared_dash_blocks]
        text+=(" · 冲刺破坏条件不成立" if any(item["identity"] in unbreakable_dash_blocks for item in pending)
               else " · 需冲刺撞开" if pending else " · 撞开条件已满足")
        if corridor["status"]!="verified":
            text+=" · 几何待验证"
    if turns>0:
        text+=f" · 第 {turns} 轮"
    reason=corridor.get("reason")
    if corridor["status"] in {"pending","trying","failed"} and isinstance(reason,str) and reason:
        reason={"no_route":"未找到可行走法","no_route_within_budget":"本次预算内未找到走法",
                "budget_exhausted":"本次搜索预算用尽","time_budget":"本次搜索时间用尽",
                "portal_unreachable":"中转位置无法到达","replay_failed":"整路模拟未通过",
                "unsupported_room_mechanics":"包含尚未支持的机关"}.get(reason,reason)
        text+="："+reason[:160]
    return text


def topology_progress_label(entries,cleared_dash_blocks=(),unbreakable_dash_blocks=()):
    """Summarize one current geometric attempt without counting public points."""
    available=[entry for entry in entries if preview_corridors(entry)]
    if not available:
        return ""
    def updated(entry):
        value=entry.get("updated_at",0)
        return value if type(value) in (int,float) and math.isfinite(value) else 0
    entry=max(available,key=updated)
    corridors=preview_corridors(entry)
    active=(entry.get("topology") or {}).get("active_id")
    index=next((i for i,item in enumerate(corridors) if item["id"]==active),None)
    if index is None:
        index=next((i for i,item in enumerate(corridors) if item["status"]=="trying"),None)
    if index is None:
        index=next((i for i,item in enumerate(corridors) if item["status"]=="verified"),0)
    text=corridor_caption(corridors[index],index+1,len(corridors),cleared_dash_blocks,unbreakable_dash_blocks)
    tried=sum((type(item.get("turns")) is int and item["turns"]>0)
              or item["status"]!="pending" for item in corridors)
    failed=sum(item["status"]=="failed" for item in corridors)
    return text+f" · 已尝试 {tried}/{len(corridors)} 条"+(f" · {failed} 条未通过" if failed else "")


def corridor_label_indices(corridors,active_id,limit=6):
    """Draw every coarse path; reserve readable labels for the current work.

    Full per-route status remains in the tooltip. Twenty-four labels at the
    same exit would hide each other and the underlying map.
    """
    order=sorted(range(len(corridors)),key=lambda index:(
        corridors[index]["id"]!=active_id,
        corridors[index]["status"]!="pending",index))
    return set(order[:limit])


def search_display(stats,labels):
    """Compact portfolio feedback with a complete, separately displayed list."""
    if not isinstance(stats,dict) or not stats.get("total"):
        return "","",""
    counter=f"策略 {stats.get('active',0)} 运行 / {stats.get('completed',0)} 完成 / {stats['total']} 总计"
    names=list(dict.fromkeys(stats.get("strategies") or []))
    active=list(dict.fromkeys(stats.get("active_strategies") or []))
    shown=active or names
    rendered=[labels.get(name,name) for name in shown]
    preview=" / ".join(rendered[:3])
    if len(rendered)>3:
        preview+=f" 等 {len(rendered)} 种"
    if preview:
        preview=("正在计算：" if active else "搜索策略：")+preview+"（悬停查看全部）"
    all_names=list(dict.fromkeys(names+active))
    complete=counter
    if active:
        complete+="\n正在运行："+" / ".join(labels.get(name,name) for name in active)
    if all_names:
        complete+="\n全部策略：\n"+"\n".join(f"{index+1}. {labels.get(name,name)}" for index,name in enumerate(all_names))
    return counter,preview,complete


class HoverInfo:
    """Optional detail on demand, without growing the map/editor layout."""
    def __init__(self,widget,text):
        self.widget,self.text=widget,text
        self.pending=None
        self.popup=None
        widget.bind("<Enter>",self.schedule,add="+")
        widget.bind("<Leave>",self.hide,add="+")
        widget.bind("<ButtonPress>",self.hide,add="+")

    def schedule(self,event=None):
        self.hide()
        self.pending=self.widget.after(400,self.show)

    def show(self):
        self.pending=None
        text=self.text()
        if not text:
            return
        popup=self.popup=tk.Toplevel(self.widget)
        popup.withdraw()
        popup.overrideredirect(True)
        tk.Label(popup,text=text,justify="left",bg="#0f172a",fg="#e2e8f0",padx=12,pady=10,
                 wraplength=520,font=("Microsoft YaHei UI",10),relief="solid",borderwidth=1).pack()
        popup.update_idletasks()
        x=min(self.widget.winfo_rootx()+10,self.widget.winfo_screenwidth()-popup.winfo_reqwidth()-12)
        y=self.widget.winfo_rooty()+self.widget.winfo_height()+4
        if y+popup.winfo_reqheight()>self.widget.winfo_screenheight()-12:
            y=self.widget.winfo_rooty()-popup.winfo_reqheight()-4
        popup.geometry(f"+{max(0,x)}+{max(0,y)}")
        popup.deiconify()

    def hide(self,event=None):
        if self.pending:
            self.widget.after_cancel(self.pending)
            self.pending=None
        if self.popup:
            self.popup.destroy()
            self.popup=None


@dataclass
class RouteDraft:
    """Editable UI data; submitted routes are detached copies, never live aliases."""
    waypoints: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    selected: int = -1

    def append(self,target):
        self.waypoints.append(deepcopy(target))
        self.segments.append(dict(no_dash=False,no_ground=False))
        self.selected=len(self.waypoints)-1

    def remove(self):
        if 0<=self.selected<len(self.waypoints):
            self.waypoints.pop(self.selected)
            self.segments.pop(self.selected)
            self.selected=min(self.selected,len(self.waypoints)-1)

    def move(self,delta):
        destination=self.selected+delta
        if 0<=self.selected<len(self.waypoints) and 0<=destination<len(self.waypoints):
            for values in (self.waypoints,self.segments):
                values[self.selected],values[destination]=values[destination],values[self.selected]
            self.selected=destination

    def clear(self):
        self.waypoints.clear()
        self.segments.clear()
        self.selected=-1

    def set_constraints(self,no_dash,no_ground):
        if 0<=self.selected<len(self.segments):
            self.segments[self.selected]=dict(no_dash=bool(no_dash),no_ground=bool(no_ground))

    def payload(self):
        return deepcopy(dict(waypoints=self.waypoints,segments=self.segments))


class AssistWindow:
    def __init__(self,root,shared,args):
        self.root,self.shared,self.args=root,shared,args
        self.map=None
        self.room=None
        self.draft=RouteDraft()
        self.exits=[]
        self.scale=2
        self.offset=(20,20)
        self.last_map_path=None
        self.last_room_name=None
        self.configuration=None
        self.game_process=None
        self.last_save=0
        self.strategy_labels=None
        self.search_tooltip=""
        self.preview_styles=PreviewStyles()
        self.preview_label_font=tkfont.Font(root=root,family="Segoe UI",size=9,weight="bold")
        self.preview_filter_key=None
        self.preview_choices={"全部":None}
        self.preview_last_draw=0
        self.preview_signature=None
        self.preview_tooltip=""
        self.preview_visible_workers=0
        self._closed=False
        self._falling_after=None
        self._falling_signature=None
        root.title("Celeste 辅助机")
        root.geometry("1100x760")
        root.minsize(880,620)
        root.configure(bg="#111827")
        style=ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",background="#111827")
        style.configure("TLabel",background="#111827",foreground="#e5e7eb",font=("Microsoft YaHei UI",10))
        style.configure("Title.TLabel",font=("Microsoft YaHei UI",17,"bold"),foreground="#f8fafc")
        style.configure("Muted.TLabel",foreground="#94a3b8")
        style.configure("TCheckbutton",background="#111827",foreground="#e5e7eb",font=("Microsoft YaHei UI",10))
        style.map("TCheckbutton",background=[("active","#1e293b")],foreground=[("disabled","#64748b")])
        style.configure("TButton",font=("Microsoft YaHei UI",10),padding=(12,7))
        style.configure("Accent.TButton",background="#38bdf8",foreground="#082f49")
        style.map("Accent.TButton",background=[("active","#7dd3fc"),("disabled","#334155")])
        outer=ttk.Frame(root,padding=18)
        outer.pack(fill="both",expand=True)
        heading=ttk.Frame(outer)
        heading.pack(fill="x")
        ttk.Label(heading,text="Celeste 辅助机",style="Title.TLabel").pack(side="left")
        self.connection=tk.StringVar(value="未连接")
        self.reconnect_request=ReconnectRequest(args.game_dir)
        self.reconnect_button=ttk.Button(heading,text="重新连接",command=self.reconnect)
        self.reconnect_button.pack(side="right")
        ttk.Label(heading,textvariable=self.connection,style="Muted.TLabel").pack(side="right",padx=(0,12))
        ttk.Label(outer,text="依次添加路线点 · 确认后连续完成 · 优先少用、晚用冲刺，必要时回退搜索",style="Muted.TLabel").pack(anchor="w",pady=(5,14))
        toolbar=ttk.Frame(outer)
        toolbar.pack(fill="x",pady=(0,10))
        ttk.Label(toolbar,text="房间").pack(side="left",padx=(0,8))
        self.room_choice=tk.StringVar()
        self.rooms=ttk.Combobox(toolbar,textvariable=self.room_choice,state="readonly",width=11)
        self.rooms.pack(side="left")
        self.rooms.bind("<<ComboboxSelected>>",self.choose_room)
        ttk.Button(toolbar,text="定位玩家",command=self.locate).pack(side="left",padx=8)
        ttk.Button(toolbar,text="−",width=3,command=lambda:self.zoom(.8)).pack(side="left")
        ttk.Button(toolbar,text="＋",width=3,command=lambda:self.zoom(1.25)).pack(side="left",padx=4)
        self.show_previews=tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar,text="通路与搜索轨迹",variable=self.show_previews,
                        command=self.preview_filter_changed).pack(side="left",padx=(10,5))
        self.preview_choice=tk.StringVar(value="全部")
        self.preview_selector=ttk.Combobox(toolbar,textvariable=self.preview_choice,
                                          values=("全部",),state="readonly",width=14)
        self.preview_selector.pack(side="left")
        self.preview_selector.bind("<<ComboboxSelected>>",self.preview_filter_changed)
        self.launch_button=ttk.Button(toolbar,text="启动游戏",command=self.launch_game)
        self.launch_button.pack(side="right")
        content=ttk.Frame(outer)
        content.pack(fill="both",expand=True)
        editor=ttk.Frame(content,width=278,padding=(14,0,0,0))
        editor.pack(side="right",fill="y")
        editor.pack_propagate(False)
        ttk.Label(editor,text="路线点（依次经过）").pack(anchor="w",pady=(0,7))
        # Reserve the fixed controls before giving the remaining height to the
        # scrollable list. Otherwise Tk can clip late-packed help/checkboxes.
        editor_controls=ttk.Frame(editor)
        editor_controls.pack(side="bottom",fill="x")
        list_frame=ttk.Frame(editor)
        list_frame.pack(fill="both",expand=True)
        self.waypoint_list=tk.Listbox(list_frame,bg="#0b1220",fg="#e5e7eb",selectbackground="#155e75",
                                     selectforeground="#ecfeff",exportselection=False,highlightthickness=1,
                                     highlightbackground="#334155",font=("Microsoft YaHei UI",10),height=5)
        self.waypoint_list.pack(side="left",fill="both",expand=True)
        scrollbar=ttk.Scrollbar(list_frame,orient="vertical",command=self.waypoint_list.yview)
        scrollbar.pack(side="right",fill="y")
        self.waypoint_list.configure(yscrollcommand=scrollbar.set)
        self.waypoint_list.bind("<<ListboxSelect>>",self.select_waypoint)
        self.waypoint_list.bind("<Delete>",lambda e:self.remove_waypoint())
        edit_buttons=ttk.Frame(editor_controls)
        edit_buttons.pack(fill="x",pady=(8,9))
        self.up_button=ttk.Button(edit_buttons,text="↑",width=2,command=lambda:self.move_waypoint(-1))
        self.up_button.pack(side="left")
        self.down_button=ttk.Button(edit_buttons,text="↓",width=2,command=lambda:self.move_waypoint(1))
        self.down_button.pack(side="left",padx=3)
        self.remove_button=ttk.Button(edit_buttons,text="删除",width=4,command=self.remove_waypoint)
        self.remove_button.pack(side="left")
        self.clear_button=ttk.Button(edit_buttons,text="清空",width=4,command=self.clear_waypoints)
        self.clear_button.pack(side="right")
        self.segment_label=tk.StringVar(value="选中路线点以设置进入该点的限制")
        ttk.Label(editor_controls,textvariable=self.segment_label,wraplength=255).pack(anchor="w",pady=(0,7))
        self.no_dash=tk.BooleanVar(value=False)
        self.no_ground=tk.BooleanVar(value=False)
        self.no_dash_check=ttk.Checkbutton(editor_controls,text="禁止冲刺",variable=self.no_dash,command=self.edit_constraints)
        self.no_dash_check.pack(anchor="w",pady=3)
        self.no_ground_check=ttk.Checkbutton(editor_controls,text="禁止落地",variable=self.no_ground,command=self.edit_constraints)
        self.no_ground_check.pack(anchor="w",pady=3)
        ttk.Label(editor_controls,text="禁止落地：可从地面起跳；离地后不能落地或沿地面到达目标。",
                  style="Muted.TLabel",wraplength=250).pack(anchor="w",pady=(7,0))
        canvas_frame=ttk.Frame(content)
        canvas_frame.pack(side="left",fill="both",expand=True)
        self.canvas=tk.Canvas(canvas_frame,bg="#0b1220",highlightthickness=1,highlightbackground="#334155")
        self.canvas.pack(fill="both",expand=True)
        self.canvas.bind("<Button-1>",self.select_tile)
        self.canvas.bind("<ButtonPress-2>",lambda e:self.canvas.scan_mark(e.x,e.y))
        self.canvas.bind("<B2-Motion>",lambda e:self.canvas.scan_dragto(e.x,e.y,gain=1))
        self.canvas.bind("<MouseWheel>",lambda e:self.zoom(1.15 if e.delta>0 else 1/1.15))
        self.canvas.bind("<Configure>",lambda e:self.draw_map(fit=True))
        ttk.Label(outer,text="蓝点：玩家   数字：路线点   青箭头：出口   灰点线：几何通路   黄点线：需冲刺撞开   彩线：物理模拟轨迹",style="Muted.TLabel").pack(anchor="w",pady=(8,8))
        controls=ttk.Frame(outer)
        controls.pack(fill="x")
        self.selection=tk.StringVar(value="点击空格或边界箭头，依次添加路线点")
        ttk.Label(controls,textvariable=self.selection).pack(side="left")
        self.cancel_button=ttk.Button(controls,text="取消辅助",command=self.cancel)
        self.cancel_button.pack(side="right")
        self.confirm_button=ttk.Button(controls,text="确认整条路线",style="Accent.TButton",command=self.confirm)
        self.confirm_button.pack(side="right",padx=8)
        self.status=tk.StringVar(value="等待游戏连接")
        self.status_label=ttk.Label(outer,textvariable=self.status,wraplength=790)
        self.status_label.pack(anchor="w",fill="x",pady=(13,0))
        self.progress=tk.StringVar(value="")
        ttk.Label(outer,textvariable=self.progress,style="Muted.TLabel").pack(anchor="w",pady=(4,0))
        self.detail=tk.StringVar(value=DRAFT_HINT+"\n"+CONTROL_HINT)
        self.detail_label=ttk.Label(outer,textvariable=self.detail,style="Muted.TLabel",wraplength=790)
        self.detail_label.pack(anchor="w",pady=(5,0))
        self.search_hover=HoverInfo(self.detail_label,lambda:self.search_tooltip)
        root.protocol("WM_DELETE_WINDOW",self.close)
        if sys.platform=="darwin":
            # Tk's default Cmd-Q exits the app before the service can cancel
            # workers. Use the same orderly close path as the window button.
            root.createcommand("tk::mac::Quit",self.close)
        root.bind("<Escape>",lambda e:self.cancel())
        self.refresh_editor()
        default_map=Path(args.game_dir)/"Content/Maps/1-ForsakenCity.bin"
        if default_map.exists():
            self.set_map(load_map(default_map))
        self.refresh()
        self.refresh_falling()

    def set_map(self,data):
        self.map=data
        self.rooms["values"]=list(data.rooms)
        self.room=data.start_room()
        self.room_choice.set(self.room.name)
        self.draft.clear()
        self.refresh_editor()
        self.draw_map(fit=True)

    def choose_room(self,event=None):
        if self.map and self.room_choice.get() in self.map.rooms:
            self.room=self.map.rooms[self.room_choice.get()]
            self.draw_map(fit=True)

    def locate(self):
        state=self.shared.snapshot()
        observed=state.get("observation") or {}
        if self.map and observed.get("room") in self.map.rooms:
            self.room=self.map.rooms[observed["room"]]
            self.room_choice.set(self.room.name)
            self.draw_map(fit=True)

    def point(self,x,y):
        return self.offset[0]+(x-self.room.x)*self.scale,self.offset[1]+(y-self.room.y)*self.scale

    def rectangle(self,x,y,w,h,**kwargs):
        left,top=self.point(x,y)
        self.canvas.create_rectangle(left,top,left+w*self.scale,top+h*self.scale,**kwargs)

    def draw_map(self,fit=False):
        if not self.room:
            return
        self.exits=room_exit_targets(self.map,self.room.name)
        left=min([self.room.x]+[target["x"] for target in self.exits])
        top=min([self.room.y]+[target["y"] for target in self.exits])
        right=max([self.room.x+self.room.width]+[target["x"]+8 for target in self.exits])
        bottom=max([self.room.y+self.room.height]+[target["y"]+8 for target in self.exits])
        width,height=max(200,self.canvas.winfo_width()),max(150,self.canvas.winfo_height())
        if fit:
            self.scale=max(.25,min(4,(width-40)/(right-left),(height-40)/(bottom-top)))
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
        margin_x=max(20,(width-(right-left)*self.scale)/2)
        margin_y=max(20,(height-(bottom-top)*self.scale)/2)
        self.offset=(margin_x+(self.room.x-left)*self.scale,
                     margin_y+(self.room.y-top)*self.scale)
        self.canvas.delete("all")
        self._falling_signature=None
        self.canvas.configure(scrollregion=(0,0,max(width,(right-left)*self.scale+40),
                                           max(height,(bottom-top)*self.scale+40)))
        self.rectangle(self.room.x,self.room.y,self.room.width,self.room.height,fill="#111d30",outline="#475569",tags="base")
        for ty,row in enumerate(self.room.solids):
            for tx,tile in enumerate(row):
                if tile not in {"0"," "}:
                    self.rectangle(self.room.x+tx*8,self.room.y+ty*8,8,8,fill="#42536a",outline="#34455d",tags="base")
        if self.scale>=1.5:
            for x in range(0,self.room.width+1,8):
                a,b=self.point(self.room.x+x,self.room.y)
                self.canvas.create_line(a,b,a,b+self.room.height*self.scale,fill="#1c2a3e",tags="grid")
            for y in range(0,self.room.height+1,8):
                a,b=self.point(self.room.x,self.room.y+y)
                self.canvas.create_line(a,b,a+self.room.width*self.scale,b,fill="#1c2a3e",tags="grid")
        hazards=[]
        moving_owners=moving_spike_owners(self.room)
        for entity in self.room.entities:
            if entity.kind.startswith("spikes") or "spinner" in entity.kind.lower():
                if (entity.kind,entity.id) not in moving_owners:
                    hazards.append(entity)
            elif entity.kind=="jumpThru":
                color="#647c91"
                self.rectangle(entity.x,entity.y,max(8,entity.width),max(3,entity.height),fill=color,outline="#91a4b7",tags="base")
            elif entity.kind in {"strawberry","refill","spring"}:
                x,y=self.point(entity.x,entity.y)
                radius=max(3,3*self.scale)
                color={"strawberry":"#f472b6","refill":"#34d399","spring":"#fbbf24"}[entity.kind]
                self.canvas.create_oval(x-radius,y-radius,x+radius,y+radius,fill=color,outline="",tags="base")
        # Platforms and hazards can share an anchor. Native upward/leftward
        # spikes extend outside that anchor; paint them after every platform so
        # entity ordering cannot erase a dangerous strip from the map.
        for entity in hazards:
            self.rectangle(*map_hazard_rect(entity),fill="#fb7185",outline="",tags="hazard")
        for target in self.exits:
            self.rectangle(target["x"],target["y"],8,8,fill="#134e4a",outline="#2dd4bf",tags="exit")
            x,y=self.point(target["x"]+4,target["y"]+4)
            arrow={"left":"←","right":"→","top":"↑","bottom":"↓"}.get(target["edge"],"→")
            self.canvas.create_text(x,y,text=arrow,fill="#99f6e4",
                                    font=("Segoe UI",max(7,int(6*self.scale)),"bold"),tags="exit")
        self.draw_search_previews(force=True)
        self.draw_dynamic()

    def preview_filter_changed(self,event=None):
        self.preview_filter_key=self.preview_choices.get(self.preview_choice.get())
        self.draw_search_previews(force=True)
        self.draw_dynamic()

    def draw_search_previews(self,state=None,force=False):
        if not self.room:
            return
        current=self.shared.snapshot() if state is None else state
        entries=current_search_previews(current)
        enabled=self.show_previews.get()
        cleared_dash_blocks=frozenset()
        unbreakable_dash_blocks=frozenset()
        if getattr(self,"map",None) is not None:
            observation=self.map_observation(current)
            geometry=moving_map_geometry(self.map,self.room.name,observation)
            cleared_dash_blocks=frozenset(item.identity for item in geometry
                if item.kind=="dashBlock" and item.known and item.rect is None)
            unbreakable_dash_blocks=frozenset(item.identity for item in geometry
                if item.kind=="dashBlock" and not item.dash_breakable)
        now=time.monotonic()
        if (not force and enabled and entries and now-self.preview_last_draw<PREVIEW_INTERVAL
                and cleared_dash_blocks==getattr(self,"preview_cleared_dash_blocks",frozenset())
                and unbreakable_dash_blocks==getattr(self,"preview_unbreakable_dash_blocks",frozenset())):
            return
        self.preview_last_draw=now
        self.preview_cleared_dash_blocks=cleared_dash_blocks
        self.preview_unbreakable_dash_blocks=unbreakable_dash_blocks
        choices={"全部":None}
        worker_details=[]
        for entry in entries:
            worker=preview_worker_key(entry)
            short,color=self.preview_styles.get(worker)
            label=entry.get("label") or (self.strategy_labels or {}).get(entry.get("strategy"),entry.get("strategy","搜索"))
            choices[f"{short} · {label}"]=worker
            local_note=" · 含局部片段" if any(isinstance(path,dict) and path.get("local_leg",entry.get("local_leg",False))
                                                   for path in entry.get("paths",[])) else ""
            worker_details.append(f"{short} · {label} · 模拟已过 {entry.get('waypoints_reached',0)} 点 · 扩展 {entry.get('expansions',0)}{local_note}")
            corridors=preview_corridors(entry)
            worker_details.extend("  "+corridor_caption(item,index+1,len(corridors),cleared_dash_blocks,unbreakable_dash_blocks)
                                  for index,item in enumerate(corridors))
        if choices!=self.preview_choices:
            self.preview_choices=choices
            self.preview_selector["values"]=list(choices)
            selected=next((text for text,key in choices.items() if key==self.preview_filter_key),"全部")
            self.preview_choice.set(selected)
            self.preview_filter_key=choices[selected]
        self.preview_tooltip=("\n灰色点线是几何通路；黄色点线需要冲刺撞开方块，仍须物理验证。\n"
                              "彩线来自实际物理模拟，可能仍是局部片段；方块只按游戏实际观测消失。\n"
                              "通路按轮次切换；状态清单保留全部候选，地图重点标注当前通路。\n"
                              "彩线是搜索片段的采样，没有显示彩线不表示尚未尝试。\n"
                              "只有完整路线验证通过并获实时准入后才会执行。\n"+"\n".join(worker_details)) if worker_details else ""
        visible=[entry for entry in entries if self.preview_filter_key is None
                 or preview_worker_key(entry)==self.preview_filter_key] if enabled else []
        self.preview_topology_status=topology_progress_label(visible,cleared_dash_blocks,unbreakable_dash_blocks)
        signature=(self.room.name,self.scale,self.offset,enabled,self.preview_filter_key,cleared_dash_blocks,unbreakable_dash_blocks,
                   tuple((preview_worker_key(entry),entry.get("generation"),entry.get("strategy"),
                          entry.get("updated_at",id(entry)),entry.get("expansions"),entry.get("topology")) for entry in visible))
        if not force and signature==self.preview_signature:
            return
        self.preview_signature=signature
        self.canvas.delete("search_preview")
        self.preview_visible_workers=0
        path_count=sum(min(2,len(entry.get("paths",[])))+len(preview_corridors(entry)) for entry in visible)
        point_budget=max(12,min(180,2400//max(1,path_count)))
        label_boxes=[]
        bounds=(self.canvas.canvasx(4),self.canvas.canvasy(4),
                self.canvas.canvasx(max(200,self.canvas.winfo_width())-4),
                self.canvas.canvasy(max(150,self.canvas.winfo_height())-4))
        for entry in visible:
            worker=preview_worker_key(entry)
            short,color=self.preview_styles.get(worker)
            endpoint=None
            endpoint_local=False
            has_path=False
            corridors=preview_corridors(entry)
            active_id=(entry.get("topology") or {}).get("active_id")
            labeled=corridor_label_indices(corridors,active_id)
            has_geometry=False
            for index,corridor in enumerate(corridors):
                conditions=[item for item in dash_break_conditions(corridor)
                            if item["identity"] not in cleared_dash_blocks and item["identity"] not in unbreakable_dash_blocks]
                geometry_color=DASH_CONDITION_COLOR if conditions else "#94a3b8"
                pieces=preview_room_segments(corridor["points"][:60],self.room.name,min(60,point_budget))
                geometry_end=None
                for piece in pieces:
                    coordinates=[coordinate for point in piece for coordinate in self.point(*point)]
                    if len(coordinates)>=4:
                        self.canvas.create_line(*coordinates,fill=geometry_color,
                                                width=2 if corridor["id"]==active_id else 1,
                                                dash=(1,5),tags="search_preview")
                    if piece:
                        geometry_end=piece[-1]
                        has_geometry=True
                if geometry_end is not None and index in labeled:
                    x,y=self.point(*geometry_end)
                    # Even verified topology stays a dotted geometric sketch.
                    # Only separate simulator-state paths receive solid lines.
                    text=f"{short} "+corridor_caption(corridor,index+1,len(corridors),cleared_dash_blocks,unbreakable_dash_blocks).split("：",1)[0]
                    font=getattr(self,"preview_label_font",None)
                    width=(font.measure(text) if font else sum(7 if ord(char)<128 else 12 for char in text))+4
                    height=(font.metrics("linespace") if font else 14)+2
                    label_x,label_y=place_preview_label(x,y,width,height,label_boxes,bounds)
                    self.canvas.create_text(label_x,label_y,text=text,anchor="nw",fill=geometry_color,
                                            font=font or ("Segoe UI",9,"bold"),tags="search_preview")
                if corridor["id"]==active_id:
                    for condition in conditions:
                        if condition["room"]!=self.room.name:
                            continue
                        approach=condition["approach"]
                        left,top,right,bottom=condition["rect"]
                        x,y=self.point(approach["x"],approach["y"]-5)
                        self.canvas.create_line(x,y,*self.point((left+right)/2,(top+bottom)/2),
                            fill=DASH_CONDITION_COLOR,dash=(1,5),arrow="last",width=2,tags="search_preview")
                        text="冲刺撞开此处"
                        font=getattr(self,"preview_label_font",None)
                        width=(font.measure(text) if font else 12*len(text))+4
                        height=(font.metrics("linespace") if font else 14)+2
                        label_x,label_y=place_preview_label(x,y,width,height,label_boxes,bounds)
                        self.canvas.create_text(label_x,label_y,text=text,anchor="nw",fill=DASH_CONDITION_COLOR,
                            font=font or ("Segoe UI",9,"bold"),tags="search_preview")
            for path in entry.get("paths",[])[:2]:
                if not isinstance(path,dict) or not isinstance(path.get("points"),list):
                    continue
                pieces=preview_room_segments(path["points"],self.room.name,point_budget)
                local=path.get("local_leg",entry.get("local_leg",False))
                alternative=path.get("kind")!="best"
                for piece in pieces:
                    coordinates=[coordinate for point in piece for coordinate in self.point(*point)]
                    if len(coordinates)>=4:
                        options=dict(fill=color,width=1 if alternative or local else 2,tags="search_preview")
                        if alternative or local:
                            options["dash"]=(2,5) if local else (6,4)
                        self.canvas.create_line(*coordinates,**options)
                    if piece:
                        has_path=True
                        if endpoint is None or not alternative:
                            endpoint=piece[-1]
                            endpoint_local=local
            if has_path or has_geometry:
                self.preview_visible_workers+=1
            if has_path:
                x,y=self.point(*endpoint)
                self.canvas.create_oval(x-2,y-2,x+2,y+2,fill=color,outline="",tags="search_preview")
                text=short+(" 模拟局部" if endpoint_local else " 模拟")
                font=getattr(self,"preview_label_font",None)
                width=(font.measure(text) if font else sum(7 if ord(char)<128 else 12 for char in text))+4
                height=(font.metrics("linespace") if font else 14)+2
                label_x,label_y=place_preview_label(x,y,width,height,label_boxes,bounds)
                self.canvas.create_text(label_x,label_y,text=text,anchor="nw",fill=color,
                                        font=font or ("Segoe UI",9,"bold"),tags="search_preview")
        # Preview vertices never connect to the current player or draft. Keep
        # them behind selectable numbered points and actual execution feedback.
        if self.canvas.find_withtag("hazard"):
            self.canvas.tag_lower("search_preview","hazard")
        elif self.canvas.find_withtag("dynamic"):
            self.canvas.tag_lower("search_preview","dynamic")

    def map_observation(self,current):
        """Do not reuse live positions across disconnects, maps or stalled telemetry."""
        obs=current.get("observation")
        if (not current.get("connected") or not isinstance(obs,dict)
                or (current.get("map_data") is not None and current["map_data"] is not self.map)):
            self._map_observation_token=None
            return None
        # Runtime frame identity, not player position: a stationary player can
        # observe a moving zip. Fixtures/legacy observations without frame IDs
        # fall back to immutable observation identity.
        token=(id(self.map),obs.get("room"),obs.get("spawn_id"),
               obs.get("engine_frame",obs.get("frame",id(obs))),obs.get("frame"))
        now=time.monotonic()
        if token!=getattr(self,"_map_observation_token",None):
            self._map_observation_token=token
            self._map_observation_seen=now
        if now-getattr(self,"_map_observation_seen",now)>MAP_OBSERVATION_MAX_AGE:
            return None
        return obs

    def draw_falling(self,current=None):
        """Refresh all observed mechanisms; keep the existing timer/API name."""
        if not self.room:
            return
        current=self.shared.snapshot() if current is None else current
        obs=self.map_observation(current)
        geometry=moving_map_geometry(self.map,self.room.name,obs)
        signature=(self.room.name,self.scale,self.offset,geometry)
        if signature==getattr(self,"_falling_signature",None):
            return
        self._falling_signature=signature
        self.canvas.delete("mechanism")
        for item in geometry:
            if item.rect is None:
                continue
            body_tags=("mechanism","mechanism_body",item.identity)
            hazard_tags=("mechanism","hazard",item.identity)
            if item.kind=="fallingBlock":
                body_tags+=("falling","falling_body")
                hazard_tags+=("falling",)
            elif item.kind=="cassetteBlock":
                body_tags+=("cassette",)
            self.rectangle(*item.rect,fill=item.color if item.solid else "",
                           outline=item.color if item.known else "#64748b",
                           dash=() if item.solid else (3,3),tags=body_tags)
            if not item.known:
                x,y,w,h=item.rect
                self.canvas.create_text(*self.point(x+w/2,y+h/2),text="?",fill="#94a3b8",
                                        tags=body_tags)
            elif item.solid and item.dash_breakable:
                x,y,w,h=item.rect
                self.canvas.create_text(*self.point(x+w/2,y+h/2),text="冲刺",fill="#182333",
                    font=("TkDefaultFont",max(7,min(11,int(4*self.scale)))),
                    tags=("mechanism","mechanism_label",item.identity))
            for rect in item.hazards:
                self.rectangle(*rect,fill="#fb7185" if item.known and item.solid else "",
                               outline="" if item.known and item.solid else "#fb7185",
                               dash=() if item.known and item.solid else (3,3),tags=hazard_tags)
        # Bodies belong beneath all hazards/previews. Attached hazards remain
        # above previews, and player/draft feedback retains its foreground.
        if self.canvas.find_withtag("hazard"):
            self.canvas.tag_lower("mechanism_body","hazard")
        if self.canvas.find_withtag("search_preview"):
            self.canvas.tag_lower("mechanism_body","search_preview")
        if self.canvas.find_withtag("dynamic"):
            self.canvas.tag_lower("mechanism","dynamic")

    def refresh_falling(self):
        if self._closed:
            return
        current=self.shared.snapshot()
        self.draw_falling(current)
        live=(self.room is not None and current.get("connected")
              and (current.get("observation") or {}).get("room")==self.room.name
              and any(entity.kind in MOVING_MAP_KINDS for entity in self.room.entities))
        self._falling_after=self.root.after(FALLING_REFRESH_MS if live else 250,self.refresh_falling)

    def draw_dynamic(self,current=None):
        if not self.room:
            return
        self.canvas.delete("dynamic")
        current=self.shared.snapshot() if current is None else current
        self.draw_falling(current)
        obs=self.map_observation(current) or {}
        if obs.get("room")==self.room.name:
            x,y=self.point(obs["x"],obs["y"]-5)
            radius=max(4,4*self.scale)
            self.canvas.create_oval(x-radius,y-radius,x+radius,y+radius,fill="#38bdf8",outline="#e0f2fe",width=2,tags="dynamic")
        for index,target in enumerate(self.draft.waypoints):
            if target["room"]!=self.room.name:
                continue
            selected=index==self.draft.selected
            for member in target.get("exit_region",()):
                self.rectangle(member["x"],member["y"],8,8,fill="",
                               outline="#f8fafc" if selected else "#67e8f9",
                               width=2,tags="dynamic")
            if index and self.draft.waypoints[index-1]["room"]==self.room.name:
                previous=self.draft.waypoints[index-1]
                self.canvas.create_line(*self.point(previous["x"]+4,previous["y"]+4),
                                        *self.point(target["x"]+4,target["y"]+4),fill="#475569",
                                        width=1,dash=(3,4),arrow="last",tags="dynamic")
            self.rectangle(target["x"],target["y"],8,8,fill="#164e63",outline="#f8fafc" if selected else "#67e8f9",
                           width=3 if selected else 2,tags="dynamic")
            x,y=self.point(target["x"]+4,target["y"]+4)
            self.canvas.create_text(x,y,text=str(index+1),fill="#ecfeff",
                                    font=("Segoe UI",max(8,int(5*self.scale)),"bold"),tags="dynamic")
        points=[]
        chunks=[]
        for sample in current.get("route",[])[current.get("route_index",0):]:
            if sample["room"]==self.room.name:
                points.extend(self.point(sample["x"],sample["y"]-5))
            elif points:
                chunks.append(points)
                points=[]
        chunks.append(points)
        for chunk in chunks:
            if len(chunk)>=4:
                self.canvas.create_line(*chunk,fill="#22d3ee",width=2,dash=(4,3),tags="dynamic")

    def zoom(self,factor):
        self.scale=max(.25,min(12,self.scale*factor))
        self.draw_map()

    def select_tile(self,event):
        if not self.room:
            return
        world_x=self.room.x+(self.canvas.canvasx(event.x)-self.offset[0])/self.scale
        world_y=self.room.y+(self.canvas.canvasy(event.y)-self.offset[1])/self.scale
        try:
            target=target_from_world(self.map,self.room.name,world_x,world_y)
            if target.get("transition_to"):
                target=expand_exit_target(self.map,target)
        except AssistValidationError:
            self.selection.set("该格无法选中；请点击空格或青色出口箭头")
        else:
            self.draft.append(target)
            self.refresh_editor()
        self.draw_dynamic()

    def refresh_editor(self):
        self.waypoint_list.delete(0,"end")
        for index,(target,segment) in enumerate(zip(self.draft.waypoints,self.draft.segments)):
            room=self.map.rooms[target["room"]]
            tx,ty=(target["x"]-room.x)//8,(target["y"]-room.y)//8
            suffix=f" → {target['transition_to']}" if target.get("transition_to") else ""
            if target.get("exit_region"):
                cells={(member["x"],member["y"]) for member in target["exit_region"]}
                suffix+=f" · 出口 {len(cells)} 格均可"
            flags=(" 禁冲" if segment["no_dash"] else "")+(" 禁落" if segment["no_ground"] else "")
            self.waypoint_list.insert("end",f"{index+1}. {room.name} ({tx}, {ty}){suffix}{flags}")
        selected=self.draft.selected
        if selected>=0:
            self.waypoint_list.selection_set(selected)
            self.waypoint_list.see(selected)
            segment=self.draft.segments[selected]
            self.no_dash.set(segment["no_dash"])
            self.no_ground.set(segment["no_ground"])
            origin="当前玩家" if selected==0 else str(selected)
            self.segment_label.set(f"第 {selected+1} 段：{origin} → {selected+1}")
            self.selection.set(f"草稿 {len(self.draft.waypoints)} 个路线点 · 已选第 {selected+1} 点")
        else:
            self.no_dash.set(False)
            self.no_ground.set(False)
            self.segment_label.set("选中路线点以设置进入该点的限制")
            self.selection.set("点击空格或边界箭头，依次添加路线点")
        for widget in (self.no_dash_check,self.no_ground_check,self.remove_button):
            widget.configure(state="normal" if selected>=0 else "disabled")
        self.up_button.configure(state="normal" if selected>0 else "disabled")
        self.down_button.configure(state="normal" if 0<=selected<len(self.draft.waypoints)-1 else "disabled")
        self.clear_button.configure(state="normal" if self.draft.waypoints else "disabled")

    def select_waypoint(self,event=None):
        selected=self.waypoint_list.curselection()
        if not selected or selected[0]==self.draft.selected:
            return
        self.draft.selected=selected[0]
        target=self.draft.waypoints[self.draft.selected]
        if self.room.name!=target["room"]:
            self.room=self.map.rooms[target["room"]]
            self.room_choice.set(self.room.name)
            self.draw_map(fit=True)
        self.refresh_editor()
        self.draw_dynamic()

    def edit_constraints(self):
        self.draft.set_constraints(self.no_dash.get(),self.no_ground.get())
        self.refresh_editor()

    def remove_waypoint(self):
        self.draft.remove()
        self.refresh_editor()
        self.draw_dynamic()

    def clear_waypoints(self):
        self.draft.clear()
        self.refresh_editor()
        self.draw_dynamic()

    def move_waypoint(self,delta):
        self.draft.move(delta)
        self.refresh_editor()
        self.draw_dynamic()

    def confirm(self):
        if self.draft.waypoints and self.shared.snapshot().get("connected"):
            self.shared.commands.put(dict(kind="route",route=self.draft.payload()))
            self.status.set("整条路线已确认，等待当前游戏状态")

    def cancel(self):
        self.shared.commands.put(dict(kind="cancel"))
        self.status.set("正在取消辅助")

    def reconnect(self):
        if self.shared.snapshot().get("connected"):
            self.status.set("当前已连接游戏，无需重新连接")
            return
        try:
            if self.reconnect_request.request():
                self.shared.update(message="正在重新连接游戏…")
                self.reconnect_button.configure(state="disabled",text="正在连接…")
                self.status.set("正在重新连接游戏…")
        except (OSError,ValueError) as error:
            self.shared.update(message=f"无法请求重新连接：{error}")
            self.status.set(f"无法请求重新连接：{error}")

    def launch_game(self):
        if self.shared.snapshot().get("connected"):
            self.status.set("已连接运行中的游戏，无需再次启动")
            return
        if self.game_process and self.game_process.poll() is None:
            self.status.set("游戏已启动，进入关卡后会自动连接")
            return
        configuration=None
        try:
            install_bridge(Path(self.args.game_dir))
            configuration=configure_game(Path(self.args.game_dir),self.args.port)
            self.configuration=configuration
            config_path=self.configuration[0]
            backup=config_path.with_name(config_path.name+".before-assist")
            if not backup.exists() and self.configuration[1] is not None:
                backup.write_bytes(self.configuration[1])
            config=ET.fromstring(config_path.read_bytes())
            field=config.find("AssistMode")
            if field is None:
                field=ET.SubElement(config,"AssistMode")
            field.text="true"
            config_path.write_bytes(ET.tostring(config,encoding="utf-8",xml_declaration=True))
            self.game_process=launch_game_process(self.args.game_dir,
                fna3d_driver=getattr(self.args,"fna3d_driver",None),
                sdl_videodriver=getattr(self.args,"sdl_videodriver",None))
            self.shared.update(message="已启动正常游戏；选择存档进入关卡后会自动连接")
        except (OSError,ValueError,ET.ParseError) as error:
            if configuration is not None:
                restore_configuration(configuration)
                self.configuration=None
            self.shared.update(message=f"启动失败：{error}")

    def refresh(self):
        state=self.shared.snapshot()
        reconnect=getattr(self,"reconnect_request",None)
        if reconnect is not None:
            message=reconnect.poll(bool(state.get("connected")))
            if message is not None:
                self.shared.update(message=message)
                state["message"]=message
            self.reconnect_button.configure(
                text="正在连接…" if reconnect.pending else "重新连接",
                state="disabled" if reconnect.pending or state.get("connected") else "normal")
        path=state.get("map_path")
        if path and path!=self.last_map_path:
            self.last_map_path=path
            self.set_map(state["map_data"])
            self.locate()
        obs=state.get("observation") or {}
        self.connection.set(f"已连接 · 房间 {obs.get('room','—')}" if state.get("connected") else "未连接 · 手动控制")
        self.status.set(state.get("message","等待连接"))
        active=state.get("mode") in {"waiting","planning","executing"}
        self.confirm_button.configure(state="normal" if self.draft.waypoints and state.get("connected") else "disabled")
        self.cancel_button.configure(state="normal" if active else "disabled")
        color="#fca5a5" if state.get("mode")=="failed" else "#86efac" if state.get("mode")=="success" else "#e5e7eb"
        self.status_label.configure(foreground=color)
        total=len(state.get("waypoints") or [])
        route_progress=f"路线 {min(state.get('waypoint_index',0),total)} / {total} 点" if total else ""
        stats=state.get("search_stats")
        if stats and self.strategy_labels is None:
            from .assist_planning import route_strategy_specs
            self.strategy_labels={item["name"]:item["label"] for item in route_strategy_specs()}
        counter,preview,complete=search_display(stats,self.strategy_labels or {})
        self.draw_search_previews(state)
        candidates=f"本房间候选 {self.preview_visible_workers} 个任务" if self.preview_visible_workers else ""
        topology=getattr(self,"preview_topology_status","")
        self.progress.set(" · ".join(value for value in (route_progress,counter,candidates,topology) if value))
        searching=state.get("mode") in {"waiting","planning"}
        self.search_tooltip=(complete+self.preview_tooltip) if searching else ""
        if searching and self.preview_visible_workers:
            preview="灰/黄点线是几何通路，黄色需冲刺撞开；彩线是物理模拟轨迹（悬停查看验证状态）"
        self.detail.set((preview if searching and preview else DRAFT_HINT)+"\n"+CONTROL_HINT)
        self.draw_dynamic(state)
        if time.monotonic()-self.last_save>1:
            self.last_save=time.monotonic()
            output=Path(self.args.output)
            output.parent.mkdir(parents=True,exist_ok=True)
            public={key:value for key,value in state.items() if key not in {"map_data","route"}}
            temporary=output.with_name(output.name+".tmp")
            temporary.write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding="utf-8")
            temporary.replace(output)
        self.root.after(100,self.refresh)

    def close(self):
        self._closed=True
        if getattr(self,"reconnect_request",None) is not None:
            self.reconnect_request.close()
        if self._falling_after is not None:
            self.root.after_cancel(self._falling_after)
            self._falling_after=None
        self.search_hover.hide()
        self.shared.commands.put(dict(kind="cancel"))
        self.shared.stop.set()
        self.root.destroy()


def run_assist_gui(args):
    shared=SharedAssistState()
    thread=start_assist_service(shared,args.game_dir,args.port,workers=args.workers,plan_seconds=args.plan_seconds)
    root=tk.Tk()
    window=AssistWindow(root,shared,args)
    if args.launch:
        root.after(250,window.launch_game)
    root.mainloop()
    shared.stop.set()
    thread.join(timeout=2)
    return 0
