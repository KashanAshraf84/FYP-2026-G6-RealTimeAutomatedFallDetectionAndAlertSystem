"""
Multi-Person Tracker
====================
Gives every person in a camera a stable ID (P1, P2, ...) so each one gets
their own 2-second pose history and their own fall classification.

- Time-based: works at any frame rate (browser ~5 FPS, worker 5-15 FPS).
- Association: Hungarian matching on a similarity that mixes box IoU with
  centre distance relative to body size, because a falling person's box
  changes from tall to wide between frames.
- Two stages, as in ByteTrack: confident detections are matched first, then
  low-confidence detections are offered to still-unmatched tracks. A fallen
  person is often detected at confidence 0.3-0.5 (GMDCSA-24: 5th percentile
  0.30 while fallen vs 0.57 while upright) and keeps the same ID.
- Recovery stage: after a missed frame a falling person reappears up to ~1.1
  box diagonals away (95th percentile), so a recently lost person may take a
  leftover detection within RECOVER_GATE diagonals if its keypoints are
  plausible. Without it, 3 of 79 falls were lost at 5 FPS.

All thresholds were chosen on GMDCSA-24 (single-person videos) with the
regression check described in docs/MODEL_CARD.md; they have not been tuned on
real multi-person footage.
- Ghost control: on GMDCSA-24, 22.5% of frames contain a second "person"
  (hanging clothes, pillows, duplicate boxes on the real person) with low
  keypoint confidence. A new track needs a confident detection with
  confident keypoints, must be re-detected on CONFIRM_HITS frames before it
  is reported, and is never started on top of an existing person.
"""

from dataclasses import dataclass, field
from itertools import count
from typing import Callable, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ai.fallnet import PoseHistory
from ai.pose_features import to_height_units

HIGH_CONF = 0.5            # detections matched in the first stage
NEW_TRACK_CONF = 0.5       # minimum box confidence to start a track
NEW_TRACK_KPT_CONF = 0.5   # minimum mean keypoint confidence to start a track
CONFIRM_HITS = 3           # confident re-detections before a track is reported
CONFIRM_MIN_S = 0.3        # ...spanning at least this long (same behaviour at 5 and 15 FPS)
CONFIRM_WINDOW_S = 1.5     # ...within this time, otherwise the candidate is dropped
TENTATIVE_MAX_GAP_S = 0.5  # a candidate missing this long is dropped
MATCH_MIN = 0.25           # minimum similarity for a detection to continue a track
MAX_AGE_S = 2.0            # a confirmed track not seen this long is deleted
DUPLICATE_OVERLAP = 0.6    # share of a new box already covered by a tracked person
CENTRE_GATE = 0.75         # centre distance (in box diagonals) where similarity reaches 0
RECOVER_GATE = 1.2         # recovery stage: search radius in box diagonals
RECOVER_MAX_S = 1.5        # recovery stage: only for people lost at most this long ago
RECOVER_KPT_CONF = 0.3     # recovery stage: ghosts that caused jumps had mean keypoint confidence 0.24-0.29


def _centres(boxes):
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def _diagonals(boxes):
    return np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])


def _intersections(a, b):
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    return w * h


def _areas(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def similarity(track_boxes, det_boxes):
    """(T,4) x (D,4) pixel boxes -> (T,D) similarity in [0,1]: mean of IoU and centre proximity."""
    inter = _intersections(track_boxes, det_boxes)
    iou = inter / (_areas(track_boxes)[:, None] + _areas(det_boxes)[None, :] - inter + 1e-9)
    dist = np.linalg.norm(_centres(track_boxes)[:, None] - _centres(det_boxes)[None, :], axis=2)
    scale = np.maximum(_diagonals(track_boxes)[:, None], _diagonals(det_boxes)[None, :]) + 1e-9
    proximity = np.clip(1.0 - dist / (CENTRE_GATE * scale), 0.0, 1.0)
    return 0.5 * iou + 0.5 * proximity


HEAD_SHOULDERS = [0, 1, 2, 3, 4, 5, 6]


def well_formed(kps, box, frame_height):
    """
    Is this detection person-like enough to start / confirm a track?

    Normally the mean confidence of all 17 keypoints must reach
    NEW_TRACK_KPT_CONF (hanging clothes and pillows score ~0.4). A person
    sitting close to a laptop webcam is cut off by the bottom of the frame,
    so legs are missing and the 17-point mean is only ~0.45; for boxes that
    touch the bottom edge, confident head + shoulders (7 points) are enough.
    Ghost detections on GMDCSA-24 do not reach the bottom edge.
    """
    c = kps[:, 2]
    if c.mean() >= NEW_TRACK_KPT_CONF:
        return True
    cut_off_at_bottom = box[3] >= 0.97 * frame_height
    return bool(cut_off_at_bottom and c[HEAD_SHOULDERS].mean() >= NEW_TRACK_KPT_CONF)


def _track_similarity(tracks, det_boxes, t):
    """Best of the motion-predicted box and the last seen box (prediction overshoots in sudden drops)."""
    predicted = np.stack([tr.predicted_box(t) for tr in tracks])
    last = np.stack([tr.box for tr in tracks])
    return np.maximum(similarity(predicted, det_boxes), similarity(last, det_boxes))


def _recovery_similarity(tracks, det_boxes, t):
    """Centre proximity within RECOVER_GATE box diagonals, for people lost at most RECOVER_MAX_S ago."""
    last = np.stack([tr.box for tr in tracks])
    dist = np.linalg.norm(_centres(last)[:, None] - _centres(det_boxes)[None, :], axis=2)
    scale = np.maximum(_diagonals(last)[:, None], _diagonals(det_boxes)[None, :]) + 1e-9
    proximity = np.clip(1.0 - dist / (RECOVER_GATE * scale), 0.0, 1.0)
    recent = np.array([t - tr.last_t <= RECOVER_MAX_S for tr in tracks])
    return proximity * recent[:, None]


def covered_fraction(box, others):
    """Share of `box` (4,) covered by each of `others` (N,4)."""
    if len(others) == 0:
        return np.zeros(0)
    return _intersections(box[None], others)[0] / (_areas(box[None])[0] + 1e-9)


@dataclass
class Track:
    box: np.ndarray                   # last matched box, pixels (x1, y1, x2, y2)
    first_t: float
    last_t: float
    track_id: Optional[str] = None    # assigned when the track is confirmed
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # box centre, px/s
    hits: int = 1
    history: PoseHistory = field(default_factory=PoseHistory)
    last_conf: float = 0.0

    @property
    def confirmed(self):
        return self.track_id is not None

    def predicted_box(self, t):
        dt = min(max(t - self.last_t, 0.0), 0.5)
        shift = self.velocity * dt
        return self.box + np.array([shift[0], shift[1], shift[0], shift[1]])


class PersonTracker:
    """Tracks every person seen by one camera."""

    def __init__(self, next_id: Optional[Callable[[], str]] = None):
        counter = count(1)
        self._next_id = next_id or (lambda: f"P{next(counter)}")
        self.tracks: List[Track] = []
        self.recovered = []

    def update(self, t, kps_px, boxes, frame_height):
        """
        Feed one frame's detections ((n,17,3) px keypoints, (n,5) boxes with
        confidence). Returns {detection index: Track} for matched or new tracks.
        """
        t = float(t)
        n = len(boxes)
        conf = boxes[:, 4] if n else np.zeros(0)
        assigned = {}
        free = list(range(len(self.tracks)))
        self.recovered = []  # (track, detection index) re-acquired by the recovery stage this frame

        def match(track_ids, det_ids, sim_fn, min_sim, confident_stage, recovery=False):
            if not track_ids or not det_ids:
                return
            sim = sim_fn([self.tracks[k] for k in track_ids], boxes[det_ids, :4], t)
            rows, cols = linear_sum_assignment(-sim)
            for r, c in zip(rows, cols):
                if sim[r, c] >= min_sim:
                    k, d = track_ids[r], det_ids[c]
                    if recovery:
                        self.recovered.append((self.tracks[k], d))
                    confident = confident_stage and well_formed(kps_px[d], boxes[d], frame_height)
                    self._apply(self.tracks[k], t, kps_px[d], boxes[d], frame_height, confident)
                    assigned[d] = self.tracks[k]
                    free.remove(k)

        confirmed = lambda: [k for k in free if self.tracks[k].confirmed]
        candidates = lambda: [k for k in free if not self.tracks[k].confirmed]
        left = lambda dets: [d for d in dets if d not in assigned]
        high = [d for d in range(n) if conf[d] >= HIGH_CONF]
        low = [d for d in range(n) if conf[d] < HIGH_CONF]

        # 1-2. Confident detections: real people first, so a ghost candidate never takes their box.
        match(confirmed(), left(high), _track_similarity, MATCH_MIN, True)
        match(candidates(), left(high), _track_similarity, MATCH_MIN, True)
        # 3. Low-confidence detections may only continue existing people (ByteTrack second stage).
        match(confirmed(), left(low), _track_similarity, MATCH_MIN, False)
        # 4. Recovery: a person lost while falling reappears up to ~1.1 box diagonals away
        #    (GMDCSA-24, 95th percentile after a missed frame); take a leftover detection nearby.
        plausible = [d for d in left(range(n)) if kps_px[d, :, 2].mean() >= RECOVER_KPT_CONF]
        match(confirmed(), plausible, _recovery_similarity, 1e-6, False, recovery=True)

        # Births: confident, well-formed detections that are not on top of someone already tracked.
        for d in left(high):
            if conf[d] < NEW_TRACK_CONF or not well_formed(kps_px[d], boxes[d], frame_height):
                continue
            occupied = np.array([tr.box for tr in self.tracks if t - tr.last_t <= TENTATIVE_MAX_GAP_S]).reshape(-1, 4)
            if (covered_fraction(boxes[d, :4], occupied) > DUPLICATE_OVERLAP).any():
                continue
            track = Track(box=boxes[d, :4].astype(float).copy(), first_t=t, last_t=t, last_conf=float(conf[d]))
            track.history.add(t, to_height_units(kps_px[d], frame_height))
            self.tracks.append(track)
            assigned[d] = track

        self._prune(t)
        return assigned

    def _apply(self, track, t, kps, box, frame_height, confident):
        dt = t - track.last_t
        if dt > 1e-3:
            centre_old = _centres(track.box[None])[0]
            centre_new = _centres(box[None, :4])[0]
            track.velocity = 0.6 * track.velocity + 0.4 * (centre_new - centre_old) / max(dt, 0.05)
        track.box = box[:4].astype(float).copy()
        track.last_t = t
        track.last_conf = float(box[4])
        track.history.add(t, to_height_units(kps, frame_height))
        if not track.confirmed and confident:
            track.hits += 1
            if track.hits >= CONFIRM_HITS and t - track.first_t >= CONFIRM_MIN_S:
                track.track_id = self._next_id()

    def _prune(self, t):
        kept = []
        for tr in self.tracks:
            if tr.confirmed:
                if t - tr.last_t <= MAX_AGE_S:
                    kept.append(tr)
            elif t - tr.last_t <= TENTATIVE_MAX_GAP_S and t - tr.first_t <= CONFIRM_WINDOW_S:
                kept.append(tr)
        self.tracks = kept

    def active(self, t, hold_s=PoseHistory.HOLD_S):
        """Confirmed tracks seen within hold_s, oldest first."""
        return sorted(
            (tr for tr in self.tracks if tr.confirmed and t - tr.last_t <= hold_s),
            key=lambda tr: tr.first_t,
        )
