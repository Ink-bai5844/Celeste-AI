"""A physical zip trigger is distinct from a later approach-height milestone."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_routes import _inventory
from celeste_search.physics import GRAB, LEFT, RIGHT, UP, SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.wall_moves import WallGripGoal
from celeste_search.zip_grips import ZipContactGoal


def fixture(mirror=False, offset=(0, 0), name="zip-event"):
    ox, oy = offset
    def px(x):
        return ox + (320-x if mirror else x)
    def left(x, width):
        return px(x+width) if mirror else px(x)
    entity = Entity("zipMover", 1, left(80, 24), oy+64, 24, 40,
                    {}, ((left(216, 24), oy+64),))
    spike = Entity("spikesUp", 2, entity.x, entity.y, 24, 0)
    room = Room(name, ox, oy, 320, 192, ("0"*40,)*24, (entity, spike), ())
    simulator = Simulator(MapData("zip-event", {name: room}, Path("fixture.bin")))
    identity = f"{name}:1"
    toward = LEFT if mirror else RIGHT
    origin = SimState(name, px(75), oy+94, vx=-90 if mirror else 90,
                      facing=-1 if mirror else 1, on_ground=False,
                      mechanisms=((identity, 0, 0, entity.x, entity.y, 0.),))
    kwargs = dict(kind="wall_grip", room=name, x=px(76), y=oy+76,
                  id="contact", identity=identity, mechanism_kind="zipMover",
                  facing=-1 if mirror else 1,
                  rect=(entity.x, entity.y, entity.x+24, entity.y+40))
    event, milestone = ZipContactGoal(**kwargs), WallGripGoal(**kwargs)
    contact = simulator.step(origin, GRAB | toward)
    return simulator, origin, contact, event, milestone, toward


class ZipContactEventTests(unittest.TestCase):
    def test_first_lower_contact_triggers_zip_in_translated_and_mirrored_physics(self):
        for mirror, offset, name in ((False, (0, 0), "original"),
                                     (False, (-720, -1200), "translated"),
                                     (True, (-480, -904), "mirror")):
            with self.subTest(room=name):
                sim, origin, contact, event, milestone, toward = fixture(mirror, offset, name)
                self.assertFalse(event.achieved(origin))
                self.assertEqual(contact, sim.step(origin, GRAB | toward))
                self.assertEqual(contact.state, 1)
                self.assertEqual(contact.mechanisms[0][1], 1)
                self.assertGreater(contact.y, event.y+4)
                self.assertTrue(event.achieved(contact))
                self.assertFalse(milestone.achieved(contact))
                public = Goal("target", name, event.x, offset[1]+16)
                inventory = _inventory(sim, contact, public)
                current = next(node.goal for node in inventory
                               if node.identity == event.identity
                               and getattr(node.goal, "facing", 0) == event.facing)
                self.assertIsInstance(current, ZipContactGoal)
                self.assertTrue(current.achieved(contact))
                self.assertFalse(public.achieved(contact))

    def test_nearby_or_invalid_states_do_not_count_as_contact(self):
        _, _, contact, event, _, _ = fixture()
        for label, state in (
            ("wrong_face", replace(contact, facing=-contact.facing)),
            ("airborne_normal", replace(contact, state=0)),
            ("no_stamina", replace(contact, stamina=0)),
            ("broken", replace(contact, broken=(event.identity,))),
            ("dead", replace(contact, dead=True)),
            ("other_room", replace(contact, room="other")),
            ("not_touching", replace(contact, x=contact.x-1)),
        ):
            with self.subTest(reason=label):
                self.assertFalse(event.achieved(state))

    def test_actual_horizontal_ride_updates_bounds_and_rejects_old_location(self):
        for mirror, offset in ((False, (0, 0)), (True, (-480, -904))):
            with self.subTest(mirror=mirror):
                sim, _, contact, event, _, toward = fixture(mirror, offset)
                state = contact
                for _ in range(32):
                    state = sim.step(state, GRAB | toward)
                    self.assertFalse(state.dead)
                    self.assertTrue(event.achieved(state))
                self.assertGreater(abs(state.x-contact.x), 16)
                self.assertNotEqual(event.bounds(state), event.bounds(contact))
                self.assertFalse(event.achieved(replace(state, x=contact.x, y=contact.y)))
                self.assertFalse(event.achieved(replace(state, mechanisms=contact.mechanisms)))

    def test_ordinary_wall_grip_still_requires_its_height(self):
        sim, _, contact, event, milestone, toward = fixture()
        self.assertTrue(event.achieved(contact))
        self.assertFalse(milestone.achieved(contact))
        state = contact
        for _ in range(40):
            state = sim.step(state, GRAB | UP | toward)
            self.assertFalse(state.dead)
            if milestone.achieved(state):
                break
        self.assertTrue(milestone.achieved(state))
        self.assertTrue(event.achieved(state))
        self.assertLessEqual(state.y, milestone.y+4)


if __name__ == "__main__":
    unittest.main()
