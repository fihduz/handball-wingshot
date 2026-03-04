import cv2
import json
import os
import numpy as np

class AreaAnnotation:
    """
    Handles calibration of reference points on the floor for pixel-to-meter mapping.
    User clicks points, enters real-world coordinates (X,Z in meters), and saves to JSON.
    Overlay draws points and labels.
    """
    def __init__(self, video_path, overlay_callback=None):
        self.video_path = video_path
        self.points = []  # [(pixel_x, pixel_y, real_x, real_z)]
        self.max_points = 8
        self.min_points = 4
        self.overlay_callback = overlay_callback
        self.homography = None
        self.calib_file = self._get_calib_file()
        self.done = False
        self._load_or_start()

    def _get_calib_file(self):
        # Save by absolute video path
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        return f"{base}_area_calib.json"

    def _load_or_start(self):
        if os.path.exists(self.calib_file):
            with open(self.calib_file, "r") as f:
                data = json.load(f)
            self.points = data["points"]
            self._compute_homography()
            self.done = True
        else:
            self.points = []
            self.done = False

    def add_point(self, pixel_x, pixel_y, real_x, real_z):
        if len(self.points) < self.max_points:
            self.points.append([pixel_x, pixel_y, real_x, real_z])
            if len(self.points) >= self.min_points:
                self._compute_homography()

    def remove_last(self):
        if self.points:
            self.points.pop()
            self.homography = None
            self.done = False

    def save(self):
        with open(self.calib_file, "w") as f:
            json.dump({"points": self.points}, f)
        self.done = True

    def _compute_homography(self):
        if len(self.points) < self.min_points:
            self.homography = None
            return
        src = np.array([[x, y] for x, y, _, _ in self.points], dtype=np.float32)
        dst = np.array([[rx, rz] for _, _, rx, rz in self.points], dtype=np.float32)
        self.homography, _ = cv2.findHomography(src, dst)

    def pixel_to_meter(self, pixel_x, pixel_y):
        if self.homography is None:
            return None
        pt = np.array([[pixel_x, pixel_y]], dtype=np.float32)
        pt = cv2.perspectiveTransform(pt[None, :, :], self.homography)[0][0]
        return pt[0], pt[1]  # (real_x, real_z)

    def draw_overlay(self, frame):
        for idx, (px, py, rx, rz) in enumerate(self.points):
            cv2.circle(frame, (int(px), int(py)), 8, (0, 255, 255), -1)
            label = f"{idx+1}: ({rx:.2f}, {rz:.2f})"
            cv2.putText(frame, label, (int(px)+10, int(py)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        if self.overlay_callback:
            self.overlay_callback(frame, self)

    def is_ready(self):
        return self.done and self.homography is not None
