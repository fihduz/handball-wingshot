import cv2
import numpy as np
import csv
import math
from pathlib import Path

from p_data import AirborneData, SpeedData


class MediaPlayerOverlay:
    """Visual overlay for the media player — shows state machine info etc."""

    def __init__(self, config):
        self.config = config
        self.airborne_data = AirborneData()
        self.speed_data = SpeedData()
        self._last_frame_shape = None
        self.floor_tracker = None
        self.length_visual = None
        self.data_window_name = "TrackingData"
        self.calibration_target = None
        self.calibration_progress = 0
        self.calibration_next_label = None
        self.calibration_ready = False
        self.player_rows = []
        self._pending_row_index = None
        self._last_selected_foot_landmark_idx = None
        self._debug_landmark_selection = True
        self.csv_enabled = bool(getattr(config, "enable_csv_export", 0))
        self.csv_path = self._init_csv_file() if self.csv_enabled else None

    def _init_csv_file(self):
        out_dir = Path("exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        video_stem = Path(self.config.video_path).stem
        csv_path = out_dir / f"tracking_{video_stem}.csv"

        # Start fresh for each run so player numbering maps to current session.
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Player",
                    "airtime (s,frames)",
                    "arm raised (a,b,c)",
                    "jumping_point (cm)",
                    "jump length (cm)",
                    "max height (cm)",
                    "strike time (s)",
                    "jump angle (deg)",
                ]
            )
        return csv_path

    def _append_csv_row(self, row):
        if not self.csv_enabled or self.csv_path is None:
            return

        def _fmt(v, digits):
            if v is None:
                return "-"
            return f"{float(v):.{digits}f}"

        def _fmt_int(v):
            if v is None:
                return "-"
            return str(int(math.ceil(float(v))))

        airtime_s = row.get("airtime_s")
        airtime_frames = row.get("frames")
        airtime_str = "-"
        if airtime_s is not None and airtime_frames is not None:
            airtime_str = f"{float(airtime_s):.3f}s,{int(airtime_frames)}f"

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    row.get("player", ""),
                    airtime_str,
                    row.get("arm_raised", "False,False,False"),
                    _fmt(row.get("jumping_point_cm"), 1),
                    _fmt(row.get("jump_length_cm"), 1),
                    row.get("max_height_cm", "-"),
                    row.get("strike_time_s", "-"),
                    _fmt_int(row.get("jump_angle_deg")),
                ]
            )

    def set_floor_tracker(self, floor_tracker):
        self.floor_tracker = floor_tracker

    def set_length_visual(self, length_visual):
        self.length_visual = length_visual

    def set_calibration_status(self, progress, target, next_label, ready):
        self.calibration_progress = progress
        self.calibration_target = target
        self.calibration_next_label = next_label
        self.calibration_ready = ready

    def toggle_speed(self):
        self.speed_data.toggle()


    def _landmark_index_to_world(self, tracked_pose, idx):
        if self.floor_tracker is None or self._last_frame_shape is None:
            return None
        if not self.floor_tracker.is_ready():
            return None
        if tracked_pose is None or tracked_pose.landmarks is None:
            return None

        frame_h, frame_w = self._last_frame_shape
        if frame_w <= 0 or frame_h <= 0:
            return None

        lm = tracked_pose.landmarks
        if idx < 0 or idx >= len(lm):
            return None

        landmark = lm[idx]
        if hasattr(landmark, "visibility") and landmark.visibility < 0.5:
            return None

        px_x = landmark.x * frame_w
        px_y = landmark.y * frame_h
        world = self.floor_tracker.pixel_to_floor(px_x, px_y)
        if world is None:
            return None

        # Per-landmark correction offsets (cm) for consistent left-foot contact point.
        offset_map = {
            31: (0,0),
            29: (10.0, -5.0),
            27: (7.5, -10.0),
        }
        off_x, off_y = offset_map.get(idx, (0.0, 0.0))
        return (float(world[0]) + off_x, float(world[1]) + off_y)

    def _get_world_point_by_priority(self, tracked_pose, indices):
        self._last_selected_foot_landmark_idx = None
        for idx in indices:
            world = self._landmark_index_to_world(tracked_pose, idx)
            if world is not None:
                self._last_selected_foot_landmark_idx = idx
                return world
        return None

    def _get_left_foot_world(self, tracked_pose, landing=False):
        if landing:
            return self._get_world_point_by_priority(tracked_pose, [31, 29, 27])
        return self._get_world_point_by_priority(tracked_pose, [31])

    def _get_reference_point_1a(self):
        if self.floor_tracker is None:
            return None
        labels = getattr(self.floor_tracker, "POINT_LABELS", None)
        real_points = getattr(self.floor_tracker, "REAL_WORLD_POINTS", None)
        if not labels or not real_points:
            return None
        try:
            idx = labels.index("1a")
        except ValueError:
            return None
        if idx >= len(real_points):
            return None
        point_1a = real_points[idx]
        return (float(point_1a[0]), float(point_1a[1]))

    @staticmethod
    def _distance_cm(p0, p1):
        if p0 is None or p1 is None:
            return None
        dx = float(p0[0]) - float(p1[0])
        dy = float(p0[1]) - float(p1[1])
        return math.hypot(dx, dy)

    def _signed_jumping_point_cm(self, takeoff_world, point_1a):
        """Signed takeoff distance to 1a: negative when takeoff is below 1a in image."""
        dist = self._distance_cm(takeoff_world, point_1a)
        if dist is None:
            return None

        if self.floor_tracker is None or not self.floor_tracker.is_ready():
            return dist

        takeoff_px = self.floor_tracker.floor_to_pixel(float(takeoff_world[0]), float(takeoff_world[1]))
        ref_px = self.floor_tracker.floor_to_pixel(float(point_1a[0]), float(point_1a[1]))
        if takeoff_px is None or ref_px is None:
            return dist

        # In image coordinates, larger y means lower on screen.
        if float(takeoff_px[1]) > float(ref_px[1]):
            return -dist
        return dist

    def toggle_airborne_timer(self, frame_index, tracked_pose):
        is_landing_toggle = self.airborne_data.active
        world_foot = self._get_left_foot_world(tracked_pose, landing=is_landing_toggle)
        selected_idx = self._last_selected_foot_landmark_idx

        # Save visual jump points slightly towards camera to compensate side-view bias.
        adjusted_world_foot = world_foot
        if self.length_visual is not None:
            adjusted_world_foot = self.length_visual.apply_towards_camera_offset(world_foot, 2.5)

        # Keep visual jump points in sync with key toggles and use them as length source.
        visual_jump_len_cm = None
        visual_jump_angle_deg = None
        if self.length_visual is not None:
            self.length_visual.update_toggle_point(adjusted_world_foot)
            visual_jump_len_cm = self.length_visual.get_current_jump_length_cm()
            visual_jump_angle_deg = self.length_visual.get_current_jump_angle_deg()

        point_1a = self._get_reference_point_1a()

        # First 3-press: show jumping point immediately as a pending row.
        if not is_landing_toggle:
            jumping_point_cm_preview = self._signed_jumping_point_cm(adjusted_world_foot, point_1a)
            pending_row = {
                "player": len(self.player_rows) + 1,
                "airtime_s": None,
                "frames": None,
                "arm_raised": "False,False,False",
                "jumping_point_cm": jumping_point_cm_preview,
                "jump_length_cm": None,
                "max_height_cm": "-",
                "strike_time_s": "-",
                "jump_angle_deg": None,
            }
            self.player_rows.append(pending_row)
            self._pending_row_index = len(self.player_rows) - 1

        record = self.airborne_data.toggle_by_frame(frame_index, self.config.fps, adjusted_world_foot, point_1a)
        if record is not None:
            jump_length_cm = visual_jump_len_cm
            if jump_length_cm is None:
                jump_length_cm = record.get("jump_len_cm")

            jump_angle_deg = visual_jump_angle_deg
            if jump_angle_deg is not None:
                jump_angle_deg = int(math.ceil(float(jump_angle_deg)))

            if self._pending_row_index is not None and 0 <= self._pending_row_index < len(self.player_rows):
                row = self.player_rows[self._pending_row_index]
                row["airtime_s"] = record.get("airtime_s")
                row["frames"] = record.get("frames")
                # Keep signed preview value from jump start (A) instead of unsigned record fallback.
                row["jumping_point_cm"] = row.get("jumping_point_cm")
                row["jump_length_cm"] = jump_length_cm
                row["jump_angle_deg"] = jump_angle_deg
                self._append_csv_row(row)
            else:
                row = {
                    "player": len(self.player_rows) + 1,
                    "airtime_s": record.get("airtime_s"),
                    "frames": record.get("frames"),
                    "arm_raised": "False,False,False",
                    "jumping_point_cm": record.get("jumping_point_cm"),
                    "jump_length_cm": jump_length_cm,
                    "max_height_cm": "-",
                    "strike_time_s": "-",
                    "jump_angle_deg": jump_angle_deg,
                }
                self.player_rows.append(row)
                self._append_csv_row(row)

            self._pending_row_index = None

            if self._debug_landmark_selection:
                print(
                    f"Jump saved | landing landmark idx: {selected_idx} | "
                    f"jumping_point_cm: {self.player_rows[-1].get('jumping_point_cm')} | "
                    f"jump_length_cm: {self.player_rows[-1].get('jump_length_cm')}"
                )
        elif self._debug_landmark_selection:
            phase = "landing" if is_landing_toggle else "start"
            print(f"Jump toggle ({phase}) | selected landmark idx: {selected_idx}")

    def draw(self, frame, tracker):
        """Render all tracking data in a separate info window."""
        # Table layout constants are defined up front so panel size can guarantee fit.
        row_h = 28
        table_x = 20
        table_y = 86

        if frame is not None:
            self._last_frame_shape = frame.shape[:2]

        tracked = None
        state_name = "IDLE"
        if tracker is not None:
            tracked = tracker.tracked_player
            state_name = tracker.state_machine.state

        state_lower = state_name.lower()

        # Update metrics per frame.
        self.speed_data.update(state_lower, tracked, self.config.fps)

        # Draw compact spreadsheet-like table.
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 1

        headers = [
            "Player",
            "airtime (s,f)",
            "arm raised (a,b,c)",
            "jumping point cm",
            "jump length cm",
            "max height cm",
            "strike time s",
            "jump angle deg",
        ]
        col_widths = [80, 130, 190, 120, 130, 130, 120, 130]
        table_w = sum(col_widths)

        panel_h = int(getattr(self.config, "data_display_height", self.config.display_height))
        min_panel_w = table_x * 2 + table_w
        configured_panel_w = int(getattr(self.config, "data_display_width", self.config.display_width))
        panel_w = max(min_panel_w, configured_panel_w)
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

        cv2.putText(panel, "Tracking Data", (20, 30), font, 0.75, (220, 220, 220), 2)

        status_color = {
            "IDLE": (200, 200, 200),
            "MOVING": (0, 255, 0),
            "AIRBORNE": (0, 165, 255),
        }.get(state_name, (255, 255, 255))
        cv2.putText(panel, f"State: {state_name}", (20, 58), font, 0.55, status_color, 1)

        # Header background
        cv2.rectangle(panel, (table_x, table_y), (table_x + table_w, table_y + row_h), (45, 45, 45), -1)

        # Vertical separators + header text
        cx = table_x
        table_bottom_y = panel_h - 10
        for i, header in enumerate(headers):
            cv2.putText(panel, header, (cx + 6, table_y + 19), font, scale, (230, 230, 230), thickness)
            cv2.line(panel, (cx, table_y), (cx, table_bottom_y), (90, 90, 90), 1)
            cx += col_widths[i]
        cv2.line(panel, (table_x + table_w, table_y), (table_x + table_w, table_bottom_y), (90, 90, 90), 1)

        # Keep table within window height
        max_rows = max(1, min(10, (panel_h - (table_y + row_h + 10)) // row_h))
        visible_rows = self.player_rows[-max_rows:]

        def fmt(v, digits=2):
            if v is None:
                return "-"
            return f"{v:.{digits}f}"

        def fmt_int(v):
            if v is None:
                return "-"
            return str(int(math.ceil(float(v))))

        for r_idx, row in enumerate(visible_rows):
            y0 = table_y + row_h * (r_idx + 1)
            y_text = y0 + 19
            cv2.line(panel, (table_x, y0), (table_x + table_w, y0), (90, 90, 90), 1)

            airtime_s = row.get("airtime_s")
            airtime_frames = row.get("frames")
            airtime_str = "-"
            if airtime_s is not None and airtime_frames is not None:
                airtime_str = f"{float(airtime_s):.3f}s,{int(airtime_frames)}f"

            values = [
                str(row.get("player", "-")),
                airtime_str,
                str(row.get("arm_raised", "False,False,False")),
                fmt(row.get("jumping_point_cm"), 1),
                fmt(row.get("jump_length_cm"), 1),
                str(row.get("max_height_cm", "-")),
                str(row.get("strike_time_s", "-")),
                fmt_int(row.get("jump_angle_deg")),
            ]

            cx = table_x
            for c_idx, value in enumerate(values):
                cv2.putText(panel, value, (cx + 6, y_text), font, scale, (220, 220, 220), thickness)
                cx += col_widths[c_idx]

        # Bottom line for table.
        bottom_y = table_y + row_h * (len(visible_rows) + 1)
        cv2.line(panel, (table_x, bottom_y), (table_x + table_w, bottom_y), (90, 90, 90), 1)

        # Calibration status under table.
        calib_y = min(panel_h - 35, bottom_y + 28)
        if self.calibration_ready:
            cv2.putText(panel, "Calibration ready", (20, calib_y), font, 0.8, (0, 255, 120), 2)
        else:
            target = self.calibration_target if self.calibration_target is not None else 0
            cv2.putText(
                panel,
                f"Calibration {self.calibration_progress}/{target}",
                (20, calib_y),
                font,
                0.8,
                (0, 255, 255),
                2,
            )
            next_label = self.calibration_next_label if self.calibration_next_label is not None else "-"
            cv2.putText(
                panel,
                f"Next: {next_label}",
                (20, calib_y + 28),
                font,
                0.65,
                (0, 255, 255),
                2,
            )

        # Draw jump map overlay at bottom of panel
        if self.length_visual is not None:
            self.length_visual.draw_on_panel(panel, self.floor_tracker)

        cv2.imshow(self.data_window_name, panel)

