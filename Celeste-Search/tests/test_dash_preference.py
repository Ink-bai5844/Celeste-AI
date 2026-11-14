from copy import deepcopy
import unittest

from celeste_search.dash_preference import dash_preference_key, dash_usage
from celeste_search.physics import DASH


def candidate(positions, charges, actions=None):
    return dict(actions=actions if actions is not None else [0]*(len(positions)-1),
                expected=[dict(room="a",x=x,y=8,dashes=dashes)
                          for x,dashes in zip(positions,charges)],anchor_state={})


class DashPreferenceTests(unittest.TestCase):
    def test_held_input_counts_real_consumption_once_and_buffered_dash_is_counted(self):
        held=candidate([0,0,0,2,4],[1,0,0,0,0],[DASH]*4)
        self.assertEqual(dash_usage(held)["dash_start_frames"],[1])
        buffered=candidate([0,0,2],[1,1,0],[0,0])
        self.assertEqual(dash_usage(buffered)["dash_start_frames"],[2])

    def test_fewer_dashes_outrank_shorter_route(self):
        walking=candidate(list(range(10)),[1]*10)
        dashing=candidate([0,2],[1,0],[DASH])
        self.assertLess(dash_preference_key(walking),dash_preference_key(dashing))

    def test_charge_refill_between_two_dashes_does_not_erase_consumption(self):
        twice=candidate([0,1,2,3,4],[1,0,1,1,0])
        once=candidate(list(range(10)),[1]*8+[0,0])
        self.assertEqual(dash_usage(twice)["dash_count"],2)
        self.assertLess(dash_preference_key(once),dash_preference_key(twice))

    def test_later_movement_progress_wins_but_stationary_padding_does_not(self):
        early=candidate([0,1,2,3,4],[1,0,0,0,0])
        late=candidate([0,1,2,3,4],[1,1,1,0,0])
        padded=candidate([0,0,0,0,1,2,3,4],[1,1,1,1,0,0,0,0])
        self.assertLess(dash_preference_key(late),dash_preference_key(early))
        self.assertLess(dash_preference_key(early),dash_preference_key(padded))

    def test_missing_charge_trace_falls_back_to_button_edges_without_mutation(self):
        data=dict(actions=[DASH,DASH,0,DASH],expected=[{}]*5,anchor_state={})
        before=deepcopy(data)
        self.assertEqual(dash_usage(data)["dash_count"],2)
        self.assertEqual(data,before)

    def test_replay_events_count_refill_and_consumption_in_same_frame(self):
        data=candidate([0,0,2],[0,0,0],[DASH,0])
        data["dash_start_frames"]=[2]
        self.assertEqual(dash_usage(data)["dash_start_frames"],[2])
        data=candidate([0,0],[1,1],[DASH])
        data["dash_start_frames"]=[1]
        self.assertEqual(dash_usage(data)["dash_count"],1)

    def test_replay_event_evidence_rejects_invalid_indices(self):
        for frames in ([0],[3],[True],[2,1],[1,1],None):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                dash_usage(dict(candidate([0,1,2],[1,1,1]),dash_start_frames=frames))


if __name__=="__main__":
    unittest.main()
