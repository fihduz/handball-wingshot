import cv2


class LengthVisual:
    """Draw a bottom-left mini-floor map with jump start/end markers and a connecting line."""

    def __init__(self, config, enabled=True):
        self.config = config
        self.enabled = enabled
        self.first_point = None
        self.second_point = None

    def update_toggle_point(self, world_point):
        """Capture floor point on each '3' toggle. Two consecutive toggles form one jump."""
        if not self.enabled or world_point is None:
            return

        if self.first_point is None:
            self.first_point = world_point
            self.second_point = None
            return

        if self.second_point is None:
            self.second_point = world_point
            return

        # Start a new pair after a completed one
        self.first_point = world_point
        self.second_point = None

    def _to_panel_px(self, point, x_min, x_max, y_min, y_max, panel_x, panel_y, panel_w, panel_h, pad):
        x, y = point
        width = max(1.0, x_max - x_min)
        height = max(1.0, y_max - y_min)

        nx = (x - x_min) / width
        ny = (y - y_min) / height

        # Keep projected points inside [0, 1] range so they stay within mini-grid.
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        px = int(panel_x + pad + nx * (panel_w - 2 * pad))
        # invert y in panel so larger y appears higher in map coordinates
        py = int(panel_y + panel_h - pad - ny * (panel_h - 2 * pad))

        # Clamp final pixels to the inner grid rectangle boundaries.
        px = max(panel_x + pad, min(panel_x + panel_w - pad, px))
        py = max(panel_y + pad, min(panel_y + panel_h - pad, py))
        return px, py

    def draw(self, frame, floor_tracker):
        if not self.enabled or floor_tracker is None:
            return
        if not floor_tracker.is_ready():
            return

        points = floor_tracker.REAL_WORLD_POINTS
        if not points:
            return

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        frame_h, frame_w = frame.shape[:2]
        panel_w = max(220, int(frame_w * 0.20))
        panel_h = max(160, int(frame_h * 0.20))
        panel_x = 15
        panel_y = frame_h - panel_h - 15
        pad = 16

        # Background panel
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (100, 100, 100),
            1,
        )

        # Visual boundary for the usable grid area.
        cv2.rectangle(
            frame,
            (panel_x + pad, panel_y + pad),
            (panel_x + panel_w - pad, panel_y + panel_h - pad),
            (80, 80, 120),
            1,
        )

        # Draw reference grid points from floor calibration model
        for ref in points:
            gx, gy = self._to_panel_px(ref, x_min, x_max, y_min, y_max, panel_x, panel_y, panel_w, panel_h, pad)
            cv2.circle(frame, (gx, gy), 2, (140, 140, 140), -1)

        # Draw captured jump points and line
        p1_px = None
        p2_px = None

        if self.first_point is not None:
            p1_px = self._to_panel_px(self.first_point, x_min, x_max, y_min, y_max, panel_x, panel_y, panel_w, panel_h, pad)
            cv2.circle(frame, p1_px, 5, (0, 200, 255), -1)
            cv2.putText(frame, "A", (p1_px[0] + 6, p1_px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        if self.second_point is not None:
            p2_px = self._to_panel_px(self.second_point, x_min, x_max, y_min, y_max, panel_x, panel_y, panel_w, panel_h, pad)
            cv2.circle(frame, p2_px, 5, (0, 255, 120), -1)
            cv2.putText(frame, "B", (p2_px[0] + 6, p2_px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)

        if p1_px is not None and p2_px is not None:
            cv2.line(frame, p1_px, p2_px, (255, 255, 255), 6)

        cv2.putText(
            frame,
            "jump map",
            (panel_x + 8, panel_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )
