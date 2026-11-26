"""
Tests for ai.tracking.PersonTracker (no model or video needed).

    .venv\\Scripts\\python -m unittest ai.test_tracking -v
"""

import unittest

import numpy as np

from ai.pose_features import partial_view
from ai.runtime import next_frame_ms
from ai.tracking import MAX_AGE_S, PersonTracker

H = 480  # frame height in pixels


def person(x1, y1, x2, y2, conf=0.9, kpt_conf=0.85):
    """One detection: (17,3) keypoints spread inside the box, and the box with confidence."""
    xs = np.linspace(x1, x2, 17)
    ys = np.linspace(y1, y2, 17)
    kps = np.stack([xs, ys, np.full(17, kpt_conf)], axis=1).astype(np.float32)
    return kps, np.array([x1, y1, x2, y2, conf], np.float32)


def feed(tracker, t, *dets):
    kps = np.stack([d[0] for d in dets]) if dets else np.zeros((0, 17, 3), np.float32)
    boxes = np.stack([d[1] for d in dets]) if dets else np.zeros((0, 5), np.float32)
    tracker.update(t, kps, boxes, H)
    return {tr.track_id: tr for tr in tracker.active(t)}


class PersonTrackerTests(unittest.TestCase):
    def test_two_people_get_separate_stable_ids(self):
        tr = PersonTracker()
        for i in range(15):  # 3 s at 5 FPS, both walking right
            t = i * 0.2
            active = feed(tr, t, person(100 + 4 * i, 100, 180 + 4 * i, 400), person(400 + 4 * i, 90, 480 + 4 * i, 400))
        self.assertEqual(sorted(active), ["P1", "P2"])
        self.assertLess(active["P1"].box[0], active["P2"].box[0])  # IDs did not swap

    def test_falling_person_keeps_id(self):
        tr = PersonTracker()
        for i in range(8):  # standing
            active = feed(tr, i * 0.2, person(300, 100, 380, 400))
        self.assertEqual(list(active), ["P1"])
        # Fall: box turns from tall to wide, drops, confidence falls to ~0.35, one frame missed.
        fall = [(290, 200, 400, 410, 0.6), None, (250, 330, 480, 430, 0.35), (240, 340, 490, 435, 0.33)]
        t = 8 * 0.2
        for box in fall:
            active = feed(tr, t, *([person(*box[:4], conf=box[4], kpt_conf=0.5)] if box else []))
            t += 0.2
        self.assertEqual(list(active), ["P1"])

    def test_weak_ghost_never_gets_an_id(self):
        tr = PersonTracker()
        for i in range(15):  # hanging clothes: confident box, poor keypoints
            active = feed(tr, i * 0.2, person(10, 20, 60, 300, conf=0.55, kpt_conf=0.3))
        self.assertEqual(active, {})

    def test_duplicate_box_on_same_person_does_not_create_second_id(self):
        tr = PersonTracker()
        for i in range(10):
            active = feed(tr, i * 0.2, person(200, 100, 300, 400), person(210, 110, 290, 300, conf=0.6))
        self.assertEqual(list(active), ["P1"])

    def test_ids_are_not_reused(self):
        tr = PersonTracker()
        for i in range(5):
            feed(tr, i * 0.2, person(100, 100, 180, 400))
        t = 1.0 + MAX_AGE_S + 0.5  # first person gone long enough to be forgotten
        for i in range(5):
            active = feed(tr, t + i * 0.2, person(400, 100, 480, 400))
        self.assertEqual(list(active), ["P2"])

    def test_history_is_per_person(self):
        tr = PersonTracker()
        for i in range(12):
            active = feed(tr, i * 0.2, person(100, 100, 180, 400), person(400, 100, 480, 400))
        p1, p2 = active["P1"], active["P2"]
        self.assertIsNotNone(p1.history.window(11 * 0.2))
        self.assertLess(p1.history.poses[-1][:, 0].mean(), p2.history.poses[-1][:, 0].mean())


class PartialViewTests(unittest.TestCase):
    def window(self, legs, upper):
        w = np.zeros((30, 17, 3), np.float32)
        w[:, :7, 2] = upper
        w[:, 11:13, 2] = 0.7   # hips are often still detected in a close-up
        w[:, 13:, 2] = legs
        return w

    def test_close_up_at_webcam_is_partial(self):
        self.assertTrue(partial_view(self.window(legs=0.03, upper=0.6), box_bottom=480, frame_height=480))

    def test_whole_body_and_fallen_person_are_not_partial(self):
        self.assertFalse(partial_view(self.window(legs=0.8, upper=0.8), box_bottom=480, frame_height=480))   # legs visible
        self.assertFalse(partial_view(self.window(legs=0.03, upper=0.6), box_bottom=300, frame_height=480))  # not cut off
        self.assertFalse(partial_view(self.window(legs=0.03, upper=0.1), box_bottom=480, frame_height=480))  # poor detection

    def test_close_up_person_is_tracked(self):
        tr = PersonTracker()
        kps = np.zeros((17, 3), np.float32)
        kps[:, 0], kps[:, 1] = 320, np.linspace(150, 479, 17)
        kps[:7, 2] = 0.7                                  # head + shoulders only
        box = np.array([150, 80, 500, 480, 0.85], np.float32)  # reaches the bottom edge
        for i in range(6):
            tr.update(i * 0.2, kps[None], box[None], H)
        self.assertEqual([p.track_id for p in tr.active(1.0)], ["P1"])


class PacingTests(unittest.TestCase):
    def test_interval_grows_with_live_cameras_and_stays_bounded(self):
        import time
        from ai import runtime

        saved = dict(runtime._last), runtime._cost
        try:
            runtime._cost = 0.03
            runtime._last.clear()
            runtime._last[1] = time.time()
            one = next_frame_ms()
            for cam in (2, 3, 4):
                runtime._last[cam] = time.time()
            four = next_frame_ms()
            runtime._cost = 5.0
            slow = next_frame_ms()
        finally:
            runtime._last.clear()
            runtime._last.update(saved[0])
            runtime._cost = saved[1]
        self.assertEqual(one, 160)            # one camera: fastest allowed rate
        self.assertGreater(four, one)         # four cameras share the budget
        self.assertLessEqual(four, 250)       # ...but each still gets at least 4 frames/s
        self.assertEqual(slow, 500)           # never slower than 2 frames/s


if __name__ == "__main__":
    unittest.main()
