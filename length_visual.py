import cv2
import math


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

    def get_current_jump_length_cm(self):
        """Return jump length in cm from current A/B world points, or None if incomplete."""
        if self.first_point is None or self.second_point is None:
            return None
        dx = float(self.second_point[0]) - float(self.first_point[0])
        dy = float(self.second_point[1]) - float(self.first_point[1])
        return math.hypot(dx, dy)

    def get_current_jump_angle_deg(self):
        """Return jump angle in degrees from A->B in floor coordinates, or None if incomplete."""
        if self.first_point is None or self.second_point is None:
            return None
        dx = float(self.second_point[0]) - float(self.first_point[0])
        dy = float(self.second_point[1]) - float(self.first_point[1])
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def apply_towards_camera_offset(world_point, offset_cm=5.0):
        """Shift point towards camera in floor space (positive y direction) by offset_cm."""
        if world_point is None:
            return None
        return (float(world_point[0]), float(world_point[1]) + float(offset_cm))

    def draw(self, frame, floor_tracker):
        """Draw debug blue dot on PoseLandmarker video frame showing exact jumping point."""
        if not self.enabled or floor_tracker is None:
            return
        if not floor_tracker.is_ready():
            return

        # Debug-only: draw a small blue dot directly in PoseLandmarker at
        # the recorded jumping point (left foot + applied offset in world-cm).
        if self.first_point is None:
            return

        px = floor_tracker.floor_to_pixel(self.first_point[0], self.first_point[1])
        if px is None:
            return

        x = int(round(px[0]))
        y = int(round(px[1]))
        cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

    def draw_on_panel(self, panel, floor_tracker=None):
        """Draw fixed 5x4 jump map graph on TrackingData panel."""
        if not self.enabled or panel is None:
            return
        if floor_tracker is None or not floor_tracker.is_ready():
            return

        points = getattr(floor_tracker, "REAL_WORLD_POINTS", None)
        if not points:
            return

        panel_h, _ = panel.shape[:2]
        panel_x = 20
        panel_y = panel_h - 112
        panel_w_box = 280
        panel_h_box = 100
        pad = 8

        # Fixed graph extents from requested layout:
        # base data: 4x3 (cols 1..4, rows a..c) at 100 cm spacing
        # padded: +1 col before 1 and +1 row before a -> 5x4 graph
        x_min = -100.0
        x_max = 300.0
        y_top = 100.0
        y_bottom = -200.0

        cv2.rectangle(panel, (panel_x, panel_y), (panel_x + panel_w_box, panel_y + panel_h_box), (40, 40, 40), -1)
        cv2.rectangle(panel, (panel_x, panel_y), (panel_x + panel_w_box, panel_y + panel_h_box), (120, 120, 120), 2)

        inner_left = panel_x + pad
        inner_top = panel_y + pad
        inner_right = panel_x + panel_w_box - pad
        inner_bottom = panel_y + panel_h_box - pad
        inner_w = max(1, inner_right - inner_left)
        inner_h = max(1, inner_bottom - inner_top)

        def world_to_panel(world_point):
            wx = float(world_point[0])
            wy = float(world_point[1])
            nx = (wx - x_min) / (x_max - x_min)
            # User-requested inverted y in map view.
            ny = (y_top - wy) / (y_top - y_bottom)
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            px = int(round(inner_left + nx * inner_w))
            py = int(round(inner_top + ny * inner_h))
            return px, py

        # Graph grid lines (5 columns x 4 rows), no labels.
        grid_color = (80, 80, 80)
        for world_x in (-100.0, 0.0, 100.0, 200.0, 300.0):
            gx, _ = world_to_panel((world_x, 0.0))
            cv2.line(panel, (gx, inner_top), (gx, inner_bottom), grid_color, 1)
        for world_y in (100.0, 0.0, -100.0, -200.0):
            _, gy = world_to_panel((0.0, world_y))
            cv2.line(panel, (inner_left, gy), (inner_right, gy), grid_color, 1)

        # Calibration points from REAL_WORLD_POINTS.
        for ref in points:
            rx, ry = world_to_panel(ref)
            cv2.circle(panel, (rx, ry), 2, (140, 140, 140), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "Jump Map", (panel_x + 6, panel_y - 4), font, 0.4, (200, 200, 200), 1)

        # 1a is world origin (0,0).
        px_1a, py_1a = world_to_panel((0.0, 0.0))
        cv2.circle(panel, (px_1a, py_1a), 3, (255, 255, 255), -1)

        # A is always jump start (first press), B is landing (second press).
        start_px = None
        landing_px = None

        if self.first_point is not None:
            start_px = world_to_panel(self.first_point)
            cv2.rectangle(panel, (start_px[0] - 5, start_px[1] - 5), (start_px[0] + 5, start_px[1] + 5), (255, 100, 0), -1)
            cv2.putText(panel, "A", (start_px[0] - 12, start_px[1] - 8), font, 0.4, (255, 100, 0), 1)

        if self.second_point is not None:
            landing_px = world_to_panel(self.second_point)
            cv2.rectangle(panel, (landing_px[0] - 5, landing_px[1] - 5), (landing_px[0] + 5, landing_px[1] + 5), (0, 255, 0), -1)
            cv2.putText(panel, "B", (landing_px[0] - 12, landing_px[1] - 8), font, 0.4, (0, 255, 0), 1)

        if start_px is not None and landing_px is not None:
            cv2.line(panel, start_px, landing_px, (200, 200, 0), 1)
