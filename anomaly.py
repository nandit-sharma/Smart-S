"""
anomaly.py  —  Motion-history accident detector  (v4)
──────────────────────────────────────────────────────
Key fixes over v3:
  • Speed is EMA-smoothed — raw bbox jitter no longer fakes motion.
  • Approach history loop fixed — samples are now in chronological order.
  • was_sustained_fast() rewritten with explicit positive indexing.
  • vehicle_bboxes in the event always carries live track IDs so main.py
    can look up the current bbox instead of using stale coordinates.
  • Collision no longer requires BOTH cars to be fast — a stationary car
    that gets hit by a fast one is still an accident.
  • DETECT_EVERY_N skipped-frame problem handled: tracker marks frames
    where detections were recycled and ignores them for velocity.

Two accident triggers
─────────────────────
1. COLLISION  – boxes overlap (IoU ≥ 0.05) AND the pair had a sustained
               high closing speed in recent history AND at least one car
               was genuinely moving before contact.

2. SUDDEN STOP – vehicle was fast for N consecutive frames then drops to
                 near-zero and holds it.  Confirms the car hit something.

show_on_screen = True only when confidence ≥ HIGH_CONF_THRESHOLD (0.94).
"""
from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Tunable constants ─────────────────────────────────────────────────────────
MAX_MATCH_DIST         = 100   # px  – re-match radius (wider = handles fast cars)
MAX_LOST_FRAMES        = 8
MIN_FRAMES_TRACKED     = 6     # frames needed before anomaly checks begin

# Collision
COLLISION_IOU_MIN      = 0.05  # genuine overlap threshold
APPROACH_WINDOW        = 5     # frames of approach-speed history to average
MIN_AVG_APPROACH       = 2.5   # px/frame – sustained closing speed
MIN_ONE_SPEED          = 3.5   # at least one car must have been this fast

# Sudden stop
SUSTAINED_N            = 5     # frames of sustained high speed required
MIN_FAST_SPEED         = 3.5   # px/frame – "fast"
STOP_SPEED_MAX         = 2.5   # px/frame – "stopped"
STOP_CONFIRM_N         = 2     # consecutive stopped frames to confirm

# Smoothing
SPEED_EMA_ALPHA        = 0.45  # EMA weight for current frame (higher = more responsive)

# Display / output
HIGH_CONF_THRESHOLD    = 0.80
COOLDOWN_SECONDS       = 4.0
ACCIDENT_HOLD_FRAMES   = 60    # frames to keep alert on-screen after event


# ── Event ─────────────────────────────────────────────────────────────────────
@dataclass
class AnomalyEvent:
    event_type:     str
    confidence:     float
    description:    str
    bbox:           Optional[list] = None
    vehicle_bboxes: list = field(default_factory=list)
    # Live track IDs so main.py can fetch current bboxes
    track_ids:      list = field(default_factory=list)
    show_on_screen: bool = False


# ── Geometry ──────────────────────────────────────────────────────────────────
def _centroid(bbox: list) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

def _dist(a: tuple, b: tuple) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

def _iou(b1: list, b2: list) -> float:
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def _merged_bbox(b1: list, b2: list) -> list:
    return [min(b1[0],b2[0]), min(b1[1],b2[1]),
            max(b1[2],b2[2]), max(b1[3],b2[3])]


# ── Per-vehicle track ─────────────────────────────────────────────────────────
class VehicleTrack:
    def __init__(self, tid: int, bbox: list, centroid: tuple):
        self.tid                  = tid
        self.bbox                 = bbox
        self.centroid             = centroid
        # Centroid history (real positions, no padding)
        self.history: deque       = deque([centroid], maxlen=20)
        # Smoothed speed at each frame (EMA)
        self.smooth_speed_history: deque = deque([0.0], maxlen=20)
        self.smooth_speed         = 0.0
        self.raw_speed            = 0.0
        self.velocity             = (0.0, 0.0)
        self.lost_frames          = 0
        self.frames_tracked       = 1
        self.stopped_frames       = 0
        # Flag: was this frame a recycled detection (YOLO didn't run)?
        self.recycled             = False

    def update(self, bbox: list, centroid: tuple, recycled: bool = False):
        """
        recycled=True when this bbox came from a previous YOLO frame.
        We still update position tracking but don't update velocity/speed
        on recycled frames to avoid fake zero-velocity from stale boxes.
        """
        self.recycled = recycled
        prev_c        = self.centroid
        self.bbox     = bbox
        self.centroid = centroid
        self.history.append(centroid)

        if not recycled:
            vx = centroid[0] - prev_c[0]
            vy = centroid[1] - prev_c[1]
            self.velocity  = (vx, vy)
            self.raw_speed = (vx**2 + vy**2) ** 0.5
            # EMA smoothing: blends current raw speed with history
            self.smooth_speed = (SPEED_EMA_ALPHA * self.raw_speed
                                 + (1 - SPEED_EMA_ALPHA) * self.smooth_speed)
        # Always record smooth speed (even on recycled frames, carry forward)
        self.smooth_speed_history.append(self.smooth_speed)

        if self.smooth_speed < STOP_SPEED_MAX:
            self.stopped_frames += 1
        else:
            self.stopped_frames = 0

        self.lost_frames    = 0
        self.frames_tracked += 1

    def recent_speeds(self, n: int) -> list[float]:
        """Return the last n smoothed speeds, oldest first."""
        h = list(self.smooth_speed_history)
        return h[-n:] if len(h) >= n else h

    def avg_recent_speed(self, n: int = 4) -> float:
        s = self.recent_speeds(n)
        return float(np.mean(s)) if s else 0.0

    def was_sustained_fast(self) -> bool:
        """
        True if, in the SUSTAINED_N frames immediately before the current
        stopped run, every frame had smooth_speed >= MIN_FAST_SPEED.
        Uses explicit positive indexing to avoid slice bugs.
        """
        h = list(self.smooth_speed_history)
        sf = self.stopped_frames
        # We need at least stopped_frames + SUSTAINED_N entries
        needed = sf + SUSTAINED_N
        if len(h) < needed:
            return False
        # The window is [-(sf+SUSTAINED_N) : -sf] (or [:] if sf==0)
        end   = len(h) - sf         # exclusive, points just before stopped run
        start = end - SUSTAINED_N   # inclusive
        if start < 0:
            return False
        window = h[start:end]
        return len(window) == SUSTAINED_N and all(s >= MIN_FAST_SPEED for s in window)


# ── Main detector ─────────────────────────────────────────────────────────────
class AnomalyDetector:

    def __init__(self):
        self._tracks:           dict[int, VehicleTrack] = {}
        self._next_id:          int   = 0
        self._frame_idx:        int   = 0
        self._last_snap:        float = 0.0
        self._held_event:       Optional[AnomalyEvent] = None
        self._hold_until_frame: int   = 0
        self._last_detection_was_fresh: bool = False

    # ── Public ────────────────────────────────────────────────────────
    def analyze(self, frame: np.ndarray,
                detections: list[dict],
                fresh_detection: bool = True) -> list[AnomalyEvent]:
        """
        fresh_detection: set False when detections are recycled (YOLO skipped).
        This prevents fake zero-velocity on non-YOLO frames.
        """
        self._frame_idx += 1
        self._last_detection_was_fresh = fresh_detection
        self._update_tracks(detections, recycled=not fresh_detection)

        events: list[AnomalyEvent] = []

        collision = self._check_collision()
        if collision:
            events.append(collision)

        stop = self._check_sudden_stop()
        if stop and not collision:
            events.append(stop)

        if events:
            best = max(events, key=lambda e: e.confidence)
            if best.show_on_screen:
                self._held_event       = best
                self._hold_until_frame = self._frame_idx + ACCIDENT_HOLD_FRAMES
            return events

        if self._held_event and self._frame_idx <= self._hold_until_frame:
            # Refresh bboxes from live tracks so boxes follow the vehicles
            live = []
            for tid in self._held_event.track_ids:
                if tid in self._tracks:
                    live.append(list(self._tracks[tid].bbox))
            if live:
                self._held_event.vehicle_bboxes = live
                if len(live) == 2:
                    self._held_event.bbox = _merged_bbox(live[0], live[1])
                else:
                    self._held_event.bbox = live[0]
            return [self._held_event]

        self._held_event = None
        return []

    def passes_snapshot_cooldown(self) -> bool:
        now = time.time()
        if now - self._last_snap >= COOLDOWN_SECONDS:
            self._last_snap = now
            return True
        return False

    def get_track_bbox(self, tid: int) -> Optional[list]:
        t = self._tracks.get(tid)
        return list(t.bbox) if t else None

    # ── Tracking ──────────────────────────────────────────────────────
    def _update_tracks(self, detections: list[dict], recycled: bool = False):
        centroids    = [_centroid(d["bbox"]) for d in detections]
        matched_tids = set()
        matched_dets = set()

        if self._tracks and detections:
            tids = list(self._tracks.keys())
            heap = []
            for ti, tid in enumerate(tids):
                tc = self._tracks[tid].centroid
                for ci, c in enumerate(centroids):
                    heapq.heappush(heap, (_dist(tc, c), ti, ci))

            while heap:
                d, ti, ci = heapq.heappop(heap)
                if d > MAX_MATCH_DIST:
                    break
                tid = tids[ti]
                if tid in matched_tids or ci in matched_dets:
                    continue
                self._tracks[tid].update(detections[ci]["bbox"], centroids[ci],
                                         recycled=recycled)
                matched_tids.add(tid)
                matched_dets.add(ci)

        # New tracks (only on fresh detections to avoid ghost tracks)
        if not recycled:
            for ci, det in enumerate(detections):
                if ci not in matched_dets:
                    tid = self._next_id
                    self._next_id += 1
                    self._tracks[tid] = VehicleTrack(tid, det["bbox"], centroids[ci])

        # Age out lost tracks
        for tid in list(self._tracks.keys()):
            if tid not in matched_tids:
                self._tracks[tid].lost_frames += 1
                if self._tracks[tid].lost_frames > MAX_LOST_FRAMES:
                    del self._tracks[tid]

    # ── Collision check ───────────────────────────────────────────────
    def _check_collision(self) -> Optional[AnomalyEvent]:
        """
        Steps:
          1. IoU ≥ COLLISION_IOU_MIN
          2. At least one vehicle had avg smooth speed ≥ MIN_ONE_SPEED
          3. The centroid-distance history shows sustained approach
             (avg closing speed ≥ MIN_AVG_APPROACH over APPROACH_WINDOW frames)
        """
        ready = [t for t in self._tracks.values()
                 if t.frames_tracked >= MIN_FRAMES_TRACKED]

        best_event = None
        best_conf  = 0.0

        for i in range(len(ready)):
            for j in range(i + 1, len(ready)):
                ti, tj = ready[i], ready[j]

                iou = _iou(ti.bbox, tj.bbox)
                if iou < COLLISION_IOU_MIN:
                    continue

                # Speed gate: at least one must have been moving
                sp_i = ti.avg_recent_speed(4)
                sp_j = tj.avg_recent_speed(4)
                if sp_i < MIN_ONE_SPEED and sp_j < MIN_ONE_SPEED:
                    continue

                # Approach speed: use centroid histories, oldest→newest
                hi = list(ti.history)  # oldest at index 0, newest at -1
                hj = list(tj.history)
                # Take the last (APPROACH_WINDOW+1) positions (oldest first)
                w = APPROACH_WINDOW + 1
                hi_w = hi[-w:] if len(hi) >= w else hi
                hj_w = hj[-w:] if len(hj) >= w else hj
                min_len = min(len(hi_w), len(hj_w))
                if min_len < 2:
                    continue

                # Pair up same-index positions (both lists oldest→newest)
                # approach[k] = dist(k) - dist(k+1): positive = closing in
                approach_samples = []
                for k in range(min_len - 1):
                    d_prev = _dist(hi_w[k],   hj_w[k])
                    d_curr = _dist(hi_w[k+1], hj_w[k+1])
                    approach_samples.append(d_prev - d_curr)

                avg_approach = float(np.mean(approach_samples))

                if avg_approach < MIN_AVG_APPROACH:
                    continue

                # Confidence — must work hard to reach 0.94:
                # Base  0.74
                # IoU   min(iou,0.50)*0.28  → max +0.14  (needs iou≥0.50 for full)
                # App   min(app,30)*0.0133  → max +0.40  (needs app≥30 px/f for full)
                # Speed min(max_sp,20)*0.005→ max +0.10
                max_sp = max(sp_i, sp_j)
                conf = round(min(0.99,
                    0.74
                    + min(iou, 0.50)      * 0.28
                    + min(avg_approach, 30) * 0.0133
                    + min(max_sp, 20)     * 0.005
                ), 2)

                if conf > best_conf:
                    best_conf  = conf
                    bboxes     = [list(ti.bbox), list(tj.bbox)]
                    best_event = AnomalyEvent(
                        event_type     = "ACCIDENT",
                        confidence     = conf,
                        description    = (
                            f"Collision IoU={iou:.3f} "
                            f"approach={avg_approach:.1f}px/f "
                            f"speeds={sp_i:.1f},{sp_j:.1f}"
                        ),
                        bbox           = _merged_bbox(ti.bbox, tj.bbox),
                        vehicle_bboxes = bboxes,
                        track_ids      = [ti.tid, tj.tid],
                        show_on_screen = conf >= HIGH_CONF_THRESHOLD,
                    )

        return best_event

    # ── Sudden stop check ─────────────────────────────────────────────
    def _check_sudden_stop(self) -> Optional[AnomalyEvent]:
        """
        Fires when:
          1. was_sustained_fast() — SUSTAINED_N consecutive fast frames before stop
          2. stopped_frames >= STOP_CONFIRM_N — truly stopped, not a blip
        """
        best_event = None
        best_conf  = 0.0

        for t in self._tracks.values():
            if t.frames_tracked < MIN_FRAMES_TRACKED:
                continue
            if t.stopped_frames < STOP_CONFIRM_N:
                continue
            if not t.was_sustained_fast():
                continue

            # Average speed in the fast window before stop
            h   = list(t.smooth_speed_history)
            sf  = t.stopped_frames
            end = len(h) - sf
            start = max(0, end - SUSTAINED_N)
            pre_window = h[start:end]
            avg_pre = float(np.mean(pre_window)) if pre_window else 0.0

            if avg_pre < MIN_FAST_SPEED:
                continue

            # Confidence: 0.94 requires avg_pre ≈ 16 px/f
            # Base 0.70 + min(avg_pre,35)*0.015
            conf = round(min(0.99, 0.70 + min(avg_pre, 35) * 0.015), 2)

            if conf > best_conf:
                best_conf  = conf
                best_event = AnomalyEvent(
                    event_type     = "ACCIDENT",
                    confidence     = conf,
                    description    = (
                        f"Sudden stop avg_pre={avg_pre:.1f}px/f "
                        f"stopped={t.stopped_frames}f "
                        f"cur={t.smooth_speed:.1f}px/f"
                    ),
                    bbox           = list(t.bbox),
                    vehicle_bboxes = [list(t.bbox)],
                    track_ids      = [t.tid],
                    show_on_screen = conf >= HIGH_CONF_THRESHOLD,
                )

        return best_event
