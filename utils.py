"""
utils.py
────────
Helpers for:
  • Snapshot saving  → alerts/
  • Output video     → recordings/
  • HUD overlay drawing
  • Accident vehicle highlighting (red boxes)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Directory setup ───────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

ALERT_DIR     = ensure_dir("alerts")
RECORDING_DIR = ensure_dir("recordings")


# ── Snapshot ──────────────────────────────────────────────────────────────────

def save_snapshot(frame: np.ndarray, event_type: str) -> str:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{event_type}_{ts}.jpg"
    filepath = ALERT_DIR / filename
    cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(filepath)


# ── Video recorder ────────────────────────────────────────────────────────────

class VideoRecorder:

    def __init__(self, video_name: str, fps: float, frame_size: tuple[int, int]):
        stem      = Path(video_name).stem
        out_path  = RECORDING_DIR / f"output_{stem}.mp4"
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        self._w   = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
        self.path = str(out_path)

    def write(self, frame: np.ndarray):
        self._w.write(frame)

    def release(self):
        self._w.release()

    def __enter__(self):  return self
    def __exit__(self, *_): self.release()


# ── Event styling ─────────────────────────────────────────────────────────────

_ACCIDENT_COLOR = (0, 0, 255)      # pure red (BGR)
_NORMAL_COLOR   = (60, 220, 60)    # green


# ── Draw red boxes on colliding vehicles ──────────────────────────────────────

def draw_accident_boxes(
    frame: np.ndarray,
    vehicle_bboxes: list[list],
) -> np.ndarray:
    """
    Draw thick red bounding boxes on each vehicle involved in the accident,
    with a bold 'ACCIDENT' label on each.
    """
    for bbox in vehicle_bboxes:
        x1, y1, x2, y2 = bbox

        # Thick red rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), _ACCIDENT_COLOR, 4)

        # Corner accents (short lines at each corner for a "targeting" look)
        L = 18  # accent length
        cv2.line(frame, (x1, y1), (x1 + L, y1), _ACCIDENT_COLOR, 5)
        cv2.line(frame, (x1, y1), (x1, y1 + L), _ACCIDENT_COLOR, 5)
        cv2.line(frame, (x2, y1), (x2 - L, y1), _ACCIDENT_COLOR, 5)
        cv2.line(frame, (x2, y1), (x2, y1 + L), _ACCIDENT_COLOR, 5)
        cv2.line(frame, (x1, y2), (x1 + L, y2), _ACCIDENT_COLOR, 5)
        cv2.line(frame, (x1, y2), (x1, y2 - L), _ACCIDENT_COLOR, 5)
        cv2.line(frame, (x2, y2), (x2 - L, y2), _ACCIDENT_COLOR, 5)
        cv2.line(frame, (x2, y2), (x2, y2 - L), _ACCIDENT_COLOR, 5)

        # Label
        label = "ACCIDENT"
        (lw, lh), _ = cv2.getTextSize(label, _FONT, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 12), (x1 + lw + 10, y1), _ACCIDENT_COLOR, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 6), _FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ── HUD overlay ───────────────────────────────────────────────────────────────

def draw_hud(
    frame:         np.ndarray,
    video_name:    str,
    fps:           float,
    vehicle_count: int,
    event_type:    str,
    confidence:    float,
    accuracy:      float,
) -> np.ndarray:
    out  = frame.copy()
    h, w = out.shape[:2]
    ts   = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # ── Info panel (top-left) ─────────────────────────────────────────
    panel_w, panel_h = 320, 188
    overlay = out.copy()
    cv2.rectangle(overlay, (6, 6), (6 + panel_w, 6 + panel_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.60, out, 0.40, 0, out)

    is_accident = (event_type != "NORMAL")
    ev_color = _ACCIDENT_COLOR if is_accident else _NORMAL_COLOR

    status_label = "ACCIDENT" if is_accident else "NORMAL"

    rows = [
        (f"Video    : {video_name}",         (18, 32),  0.60, (190, 190, 190)),
        (f"FPS      : {fps:.1f}",             (18, 58),  0.60, (170, 220, 255)),
        (f"Vehicles : {vehicle_count}",       (18, 84),  0.60, (170, 220, 255)),
        (f"Status   : {status_label}",        (18, 110), 0.60, ev_color),
        (f"Conf     : {confidence:.2f}",      (18, 136), 0.60, (190, 190, 190)),
        (f"Accuracy : {accuracy:.0f}%",       (18, 162), 0.60, (110, 255, 160)),
    ]
    for text, pos, scale, color in rows:
        cv2.putText(out, text, pos, _FONT, scale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, text, pos, _FONT, scale, color,   1, cv2.LINE_AA)

    # ── Timestamp (top-right) ─────────────────────────────────────────
    (tw, _), _ = cv2.getTextSize(ts, _FONT, 0.50, 1)
    tx = w - tw - 10
    cv2.putText(out, ts, (tx, 26), _FONT, 0.50, (0,0,0),       2, cv2.LINE_AA)
    cv2.putText(out, ts, (tx, 26), _FONT, 0.50, (210, 210, 210),1, cv2.LINE_AA)

    # ── Alert banner (bottom) ─────────────────────────────────────────
    if is_accident:
        banner = "⚠  ACCIDENT DETECTED"
        ov2    = out.copy()
        cv2.rectangle(ov2, (0, h - 54), (w, h), _ACCIDENT_COLOR, -1)
        cv2.addWeighted(ov2, 0.72, out, 0.28, 0, out)
        (bw, _), _ = cv2.getTextSize(banner, _FONT, 0.95, 2)
        bx = (w - bw) // 2
        cv2.putText(out, banner, (bx, h - 14), _FONT, 0.95, (0,0,0),     4, cv2.LINE_AA)
        cv2.putText(out, banner, (bx, h - 14), _FONT, 0.95, (255,255,255),2, cv2.LINE_AA)

    return out
