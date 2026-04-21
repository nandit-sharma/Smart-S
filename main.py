"""
main.py
───────
Smart Road Surveillance System — Entry Point  (v4)

Changes vs v3:
  • Passes fresh_detection=True/False to anomaly detector so velocity is
    only computed on real YOLO frames, not recycled ones.
  • Draws accident boxes from LIVE track bboxes (via track_ids) instead of
    stale coordinates saved at detection time — boxes now follow vehicles.
  • Screen alert (red boxes + banner) shown ONLY when conf >= 0.94.
  • Terminal logs every event regardless of confidence.

Run:
    python main.py
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import cv2

from detector import VehicleDetector
from anomaly  import AnomalyDetector
from utils    import VideoRecorder, draw_hud, draw_accident_boxes, save_snapshot, ensure_dir

# ── Settings ──────────────────────────────────────────────────────────────────
DEMO_DIR          = "demo_videos"
YOLO_MODEL        = "yolov8m.pt"
DETECT_CONFIDENCE = 0.38
DEVICE            = "cpu"           # "cuda" / "mps" for GPU

# Run YOLO every N frames; reuse boxes in between.
# Lower = more accurate tracking, higher = faster.
DETECT_EVERY_N    = 1            # 1=every frame | 2=every other | 3=every 3rd

WINDOW_NAME       = "Smart Road Surveillance"

MODEL_ACCURACY: dict[str, float] = {
    "yolov8n.pt": 86.0,
    "yolov8s.pt": 89.5,
    "yolov8m.pt": 92.3,
    "yolov8l.pt": 93.9,
    "yolov8x.pt": 95.1,
}


# ── Video selector ────────────────────────────────────────────────────────────
def pick_video(folder: str) -> str:
    videos = sorted(glob.glob(os.path.join(folder, "*.mp4")))
    if not videos:
        print(f"\n[ERROR] No .mp4 files found in '{folder}'.")
        print(f"        Place your video files there and re-run.\n")
        sys.exit(1)

    print("\n" + "=" * 56)
    print("  Smart Road Surveillance System  v4.0")
    print("=" * 56)
    print(f"\n  Videos available in '{folder}':\n")
    for idx, v in enumerate(videos, start=1):
        size_mb = Path(v).stat().st_size / 1_048_576
        print(f"    [{idx}]  {Path(v).name}  ({size_mb:.1f} MB)")

    print(f"    [0]  Process ALL videos one by one")
    print()

    while True:
        raw = input("  Enter number: ").strip()
        if raw == "0":
            return "__ALL__"
        if raw.isdigit() and 1 <= int(raw) <= len(videos):
            return videos[int(raw) - 1]
        print(f"  Please enter a number between 0 and {len(videos)}.")


# ── Per-video processing ──────────────────────────────────────────────────────
def process_video(
    video_path:     str,
    detector:       VehicleDetector,
    model_accuracy: float,
) -> None:
    video_name = Path(video_path).name
    cap        = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[WARNING] Cannot open '{video_path}' — skipping.")
        return

    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_ms = max(1, int(1000 / src_fps))

    print(f"\n{'─'*56}")
    print(f" Now playing : {video_name}")
    print(f" Resolution  : {width}×{height}  |  FPS: {src_fps:.1f}")
    print(f"{'─'*56}")
    print("  Press  Q  to quit   |   S  to skip to next video")

    anomaly_det   = AnomalyDetector()
    recorder      = VideoRecorder(video_name, fps=src_fps, frame_size=(width, height))

    fps_display   = src_fps
    frame_idx     = 0
    last_dets     = []
    t_prev        = time.perf_counter()
    blink_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t_now      = time.perf_counter()
            elapsed_s  = t_now - t_prev
            t_prev     = t_now

            # Smooth FPS display
            inst_fps    = 1.0 / max(elapsed_s, 1e-6)
            fps_display = 0.85 * fps_display + 0.15 * inst_fps

            # ── Detection ──────────────────────────────────────────────
            fresh = (frame_idx % DETECT_EVERY_N == 0)
            if fresh:
                last_dets = detector.detect(frame)
            detections = last_dets

            # ── Anomaly analysis ───────────────────────────────────────
            # Pass fresh=True only on YOLO frames so velocity is correct
            events = anomaly_det.analyze(frame, detections,
                                         fresh_detection=fresh)

            event_type     = "NORMAL"
            event_conf     = 0.0
            vehicle_bboxes = []   # live bboxes to draw
            show_on_screen = False

            if events:
                top        = max(events, key=lambda e: e.confidence)
                event_conf = top.confidence

                # Always print to terminal
                flag = "SHOW" if top.show_on_screen else f"low-conf({event_conf:.2f})"
                print(f"  ⚠  {top.description}  [{flag}]")

                # Snapshot on cooldown for high-conf events
                if top.show_on_screen and anomaly_det.passes_snapshot_cooldown():
                    snap = save_snapshot(frame, top.event_type)
                    print(f"     Snapshot → {snap}")

                if top.show_on_screen:
                    event_type = top.event_type  # "ACCIDENT"
                    show_on_screen = True

                    # ── Use LIVE bboxes from current tracks ────────────
                    # This prevents boxes being drawn at stale/wrong positions
                    live_bboxes = []
                    for tid in top.track_ids:
                        bbox = anomaly_det.get_track_bbox(tid)
                        if bbox:
                            live_bboxes.append(bbox)
                    # Fall back to event bboxes only if tracks were lost
                    vehicle_bboxes = live_bboxes if live_bboxes else top.vehicle_bboxes

            # ── Draw normal vehicle boxes ──────────────────────────────
            annotated = detector.draw(frame, detections)

            # ── Draw red accident boxes (high-conf only) ───────────────
            if show_on_screen and vehicle_bboxes:
                annotated = draw_accident_boxes(annotated, vehicle_bboxes)

            # ── Blink effect ───────────────────────────────────────────
            blink_counter += 1
            blink_active   = (event_type != "NORMAL") and (blink_counter % 8 < 4)
            hud_event_type = ">>> ACCIDENT <<<" if blink_active else event_type

            # ── HUD ────────────────────────────────────────────────────
            hud = draw_hud(
                frame         = annotated,
                video_name    = video_name,
                fps           = fps_display,
                vehicle_count = len(detections),
                event_type    = hud_event_type,
                confidence    = event_conf,
                accuracy      = model_accuracy,
            )

            # ── Record & display ───────────────────────────────────────
            recorder.write(hud)
            cv2.imshow(WINDOW_NAME, hud)

            proc_ms = int((time.perf_counter() - t_now) * 1000)
            wait_ms = max(1, frame_ms - proc_ms)
            key     = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                print("\n[INFO] Quit requested.")
                cap.release()
                recorder.release()
                cv2.destroyAllWindows()
                sys.exit(0)

            if key == ord("s"):
                print("\n[INFO] Skipping to next video.")
                break

    finally:
        cap.release()
        recorder.release()
        print(f" Saved recording → {recorder.path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ensure_dir("alerts")
    ensure_dir("recordings")

    choice = pick_video(DEMO_DIR)

    print(f"\n[INFO] Loading YOLO model: {YOLO_MODEL}  (device={DEVICE}) …")
    detector       = VehicleDetector(YOLO_MODEL, DETECT_CONFIDENCE, DEVICE)
    model_accuracy = MODEL_ACCURACY.get(YOLO_MODEL, 90.0)
    print(f"[INFO] Model ready — accuracy estimate: {model_accuracy:.1f}%\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    if choice == "__ALL__":
        videos = sorted(glob.glob(os.path.join(DEMO_DIR, "*.mp4")))
        for v in videos:
            process_video(v, detector, model_accuracy)
        print("\n[INFO] All videos processed. Done.")
    else:
        process_video(choice, detector, model_accuracy)
        print("\n[INFO] Done.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
