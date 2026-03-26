import json
from pathlib import Path

import cv2
import numpy as np


class RoomTrackingFloor:
    """Floor-plane calibration and mapping between image pixels and floor coordinates (cm)."""

    POINT_LABELS = ["1a", "1b", "1c", "1d", "2a", "2b", "2c", "3a", "3b"]
    CALIBRATION_MODEL_VERSION = 2

    # Fixed real-world coordinates (cm) in click order
    REAL_WORLD_POINTS = [
        (0, 0),      # 1a
        (100, 0),    # 1b
        (200, 0),    # 1c
        (300, 0),    # 1d
        (100, -100), # 2a
        (200, -100), # 2b
        (300, -100), # 2c
        (200, -200), # 3a
        (300, -200), # 3b
    ]

    def __init__(self, video_path):
        self.pixel_points = []
        self.homography = None
        self.ready = False
        self.calib_path = self._get_calib_path(video_path)
        self.load_calibration()

    def _get_calib_path(self, video_path):
        base_name = Path(video_path).stem
        calib_dir = Path("calibrations")
        calib_dir.mkdir(parents=True, exist_ok=True)
        return calib_dir / f"{base_name}.json"

    def point_count(self):
        return len(self.pixel_points)

    def next_point_label(self):
        idx = len(self.pixel_points)
        if idx >= len(self.POINT_LABELS):
            return None
        return self.POINT_LABELS[idx]

    def add_pixel_point(self, x, y):
        if self.ready:
            return False, "Calibration already complete"
        if len(self.pixel_points) >= len(self.REAL_WORLD_POINTS):
            return False, "All points are already added"

        self.pixel_points.append((float(x), float(y)))
        if len(self.pixel_points) == len(self.REAL_WORLD_POINTS):
            self._compute_homography()
            if self.ready:
                self.save_calibration()
                return True, "Calibration complete and saved"
            return False, "Homography failed"

        return True, f"Added point {self.POINT_LABELS[len(self.pixel_points)-1]}"

    def undo_last_point(self):
        if not self.pixel_points:
            return False, "No point to undo"
        self.pixel_points.pop()
        self.homography = None
        self.ready = False
        return True, "Last point removed"

    def reset(self):
        self.pixel_points = []
        self.homography = None
        self.ready = False
        if self.calib_path.exists():
            self.calib_path.unlink()

    def _compute_homography(self):
        src = np.array(self.pixel_points, dtype=np.float32)
        dst = np.array(self.REAL_WORLD_POINTS, dtype=np.float32)
        self.homography, _ = cv2.findHomography(src, dst)
        self.ready = self.homography is not None

    def pixel_to_floor(self, x, y):
        if not self.ready or self.homography is None:
            return None
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.homography)[0][0]
        return float(mapped[0]), float(mapped[1])

    def floor_to_pixel(self, x, y):
        if not self.ready or self.homography is None:
            return None
        h = np.asarray(self.homography, dtype=np.float64)
        inv_h = np.linalg.inv(h)
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, inv_h)[0][0]
        return float(mapped[0]), float(mapped[1])

    def is_ready(self):
        return self.ready

    def get_pixel_points(self):
        return list(self.pixel_points)

    def save_calibration(self):
        data = {
            "model_version": self.CALIBRATION_MODEL_VERSION,
            "pixel_points": self.pixel_points,
            "real_world_points": self.REAL_WORLD_POINTS,
            "labels": self.POINT_LABELS,
        }
        with self.calib_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_calibration(self):
        if not self.calib_path.exists():
            return False

        with self.calib_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Prevent silent reuse of stale calibrations from a different point model.
        loaded_version = data.get("model_version")
        loaded_labels = data.get("labels")
        loaded_world = data.get("real_world_points")
        expected_world = [list(p) for p in self.REAL_WORLD_POINTS]
        if (
            loaded_version != self.CALIBRATION_MODEL_VERSION
            or loaded_labels != self.POINT_LABELS
            or loaded_world != expected_world
        ):
            self.pixel_points = []
            self.homography = None
            self.ready = False
            return False

        loaded = data.get("pixel_points", [])
        self.pixel_points = [tuple(pt) for pt in loaded]
        if len(self.pixel_points) == len(self.REAL_WORLD_POINTS):
            self._compute_homography()
            return self.ready

        self.pixel_points = []
        self.homography = None
        self.ready = False
        return False
