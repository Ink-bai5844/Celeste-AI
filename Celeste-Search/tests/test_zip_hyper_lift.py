"""Native moving-platform launch arithmetic regression checks."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import JUMP, LEFT, RIGHT, SimState, Simulator, _zip_ease


class ZipHyperLiftTests(unittest.TestCase):
    def flat_simulator(self):
        room = Room("a", 0, 0, 320, 184,
                    ("0" * 40,) * 21 + ("1" * 40,) * 2,
                    (), ((160.0, 168.0),))
        return Simulator(MapData("flat", {"a": room}, Path("flat.bin")))

    def test_crouched_superjump_scales_lift_boost_in_both_directions(self):
        # Player.SuperJump adds LiftBoost, then applies 1.25 X / 0.5 Y.
        # The live 7a jump used capped -130 lift: (-105-130)*.5 = -117.5.
        for facing, direction in ((1, RIGHT), (-1, LEFT)):
            with self.subTest(facing=facing):
                sim = self.flat_simulator()
                start = replace(sim.initial_state(), state=2, dash_frames=4,
                                dash_dx=facing, dash_dy=0, dashes=0, ducking=True,
                                facing=facing, lift_timer=8,
                                lift_vx=40.0 * facing, lift_vy=-200.0)
                jumped = sim.step(start, direction | JUMP)
                self.assertEqual((jumped.vx, jumped.vy), (375.0 * facing, -117.5))
                self.assertEqual(jumped.var_speed, -117.5)
                self.assertFalse(jumped.ducking)
                self.assertEqual(start.lift_vy, -200.0)

    def test_standing_superjump_adds_lift_once(self):
        sim = self.flat_simulator()
        start = replace(sim.initial_state(), state=2, dash_frames=4,
                        dash_dx=1, dash_dy=0, lift_timer=8,
                        lift_vx=40, lift_vy=-200)
        jumped = sim.step(start, RIGHT | JUMP)
        self.assertEqual((jumped.vx, jumped.vy, jumped.var_speed), (300, -235, -235))

    def test_zip_math_matches_native_ease_and_xna_component_store(self):
        # Exact native values from the local x86 Ease.SineIn / Vector2.Lerp
        # calls in artifacts/zip-native-math-probe.ps1 (no game execution).
        self.assertEqual(_zip_ease(18, 30), 0.4122162461280823)
        mover = Entity("zipMover", 7, 3032, -1528, 16, 16,
                       {}, ((3032.0, -1600.0),))
        room = Room("a", 2856, -1672, 368, 216,
                    ("0" * 46,) * 27, (mover,), ((3040.0, -1552.0),))
        sim = Simulator(MapData("zip", {"a": room}, Path("zip.bin")))
        start = SimState("a", 3040, -1552, on_ground=True,
                         mechanisms=(("a:7", 2, 16, 3032.0, -1551.8226318359375, 0.0),))
        moved = sim._advance_mechanisms(start)
        self.assertEqual(moved.mechanisms[0][4], -1554.6890869140625)
        self.assertAlmostEqual(moved.lift_vy, -171.986969, places=5)
        self.assertEqual(moved.y, -1555)


if __name__ == "__main__":
    unittest.main()
