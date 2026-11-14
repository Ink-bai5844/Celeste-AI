"""Danger strips must remain visible when a later platform shares their anchor."""
from pathlib import Path
from unittest.mock import Mock
import unittest

from celeste_search.assist_gui import AssistWindow,map_hazard_rect
from celeste_search.maps import Entity,MapData,Room


class Canvas:
    def __init__(self):
        self.rectangles=[]
    def winfo_width(self):return 500
    def winfo_height(self):return 300
    def xview_moveto(self,value):pass
    def yview_moveto(self,value):pass
    def delete(self,tag):self.rectangles=[]
    def configure(self,**kwargs):pass
    def create_rectangle(self,*args,**kwargs):self.rectangles.append((args,kwargs))
    def create_line(self,*args,**kwargs):pass


class MapHazardTests(unittest.TestCase):
    def test_spike_render_bounds_follow_all_four_native_directions(self):
        expected={"spikesUp":(100,197,16,3),"spikesDown":(100,200,16,3),
                  "spikesLeft":(97,200,3,24),"spikesRight":(100,200,3,24)}
        for kind,bounds in expected.items():
            with self.subTest(kind=kind):
                self.assertEqual(map_hazard_rect(Entity(kind,1,100,200,16,24)),bounds)

    def test_later_jumpthrough_cannot_paint_over_up_spike_strip(self):
        spikes=Entity("spikesUp",8,16,32,64,0)
        platform=Entity("jumpThru",33,24,32,24,0)
        room=Room("a",0,0,128,80,("0"*16,)*10,(spikes,platform),((8.,64.),))
        window=object.__new__(AssistWindow)
        window.room=room
        window.map=MapData("fixture",{"a":room},Path("fixture.bin"))
        window.canvas=Canvas()
        window.scale=2
        window.offset=(20,20)
        window.draw_search_previews=Mock()
        window.draw_dynamic=Mock()
        window.draw_map(fit=True)
        records=window.canvas.rectangles
        hazard_index=next(i for i,(_,options) in enumerate(records) if options.get("tags")=="hazard")
        platform_index=next(i for i,(_,options) in enumerate(records) if options.get("fill")=="#647c91")
        self.assertGreater(hazard_index,platform_index)
        start=window.point(16,29)
        end=window.point(80,32)
        self.assertEqual(records[hazard_index][0],(*start,*end))
        window.draw_search_previews.assert_called_once_with(force=True)


if __name__=="__main__":
    unittest.main()
