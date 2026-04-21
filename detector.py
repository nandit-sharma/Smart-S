"""
detector.py
───────────
YOLO-based vehicle detection.
Detects: car, truck, bus, motorcycle.
Draws coloured bounding boxes with labels and confidence scores.
"""
from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class IDs for vehicles we care about
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_COLORS = {
    "car":        (0,   200, 255),   # amber
    "motorcycle": (255, 100,   0),   # blue
    "bus":        (0,   220, 100),   # green
    "truck":      (180,   0, 255),   # purple
}


class VehicleDetector:

    def __init__(self, model_name: str = "yolov8m.pt",
                 confidence: float = 0.40,
                 device: str = "cpu"):
        self.model      = YOLO(model_name)
        self.confidence = confidence
        self.device     = device
        # Warm-up pass so first-frame latency is not penalised
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, device=self.device)

    # ──────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of dicts:
            bbox     : [x1, y1, x2, y2]
            label    : str
            conf     : float
            class_id : int
        """
        results    = self.model(frame, verbose=False, device=self.device,
                                conf=self.confidence)[0]
        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox":     [x1, y1, x2, y2],
                "label":    VEHICLE_CLASSES[cls_id],
                "conf":     float(box.conf[0]),
                "class_id": cls_id,
            })

        return detections

    # ──────────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes + labels onto the frame (in-place for speed)."""
        out = frame
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label  = det["label"]
            conf   = det["conf"]
            color  = CLASS_COLORS.get(label, (180, 180, 180))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        return out
