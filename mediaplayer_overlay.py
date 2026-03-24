import cv2

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

    def set_floor_tracker(self, floor_tracker):
        self.floor_tracker = floor_tracker

    def set_length_visual(self, length_visual):
        self.length_visual = length_visual

    def toggle_speed(self):
        self.speed_data.toggle()


    def _get_left_foot_world(self, tracked_pose):
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
        world_points = []
        for idx in (27, 29, 31):
            px_x = lm[idx].x * frame_w
            px_y = lm[idx].y * frame_h
            world = self.floor_tracker.pixel_to_floor(px_x, px_y)
            if world is not None:
                world_points.append(world)

        if not world_points:
            return None

        avg_r = sum(p[0] for p in world_points) / len(world_points)
        avg_y = sum(p[1] for p in world_points) / len(world_points)
        return (avg_r, avg_y)

    def toggle_airborne_timer(self, frame_index, tracked_pose):
        world_foot = self._get_left_foot_world(tracked_pose)
        self.airborne_data.toggle_by_frame(frame_index, self.config.fps, world_foot)
        if self.length_visual is not None:
            self.length_visual.update_toggle_point(world_foot)

    def draw(self, frame, tracker):
        """Draw state machine info on the frame."""
        if tracker is None:
            return

        self._last_frame_shape = frame.shape[:2]

        state_machine = tracker.state_machine
        tracked = tracker.tracked_player
        state_lower = state_machine.state.lower()

        # Uppdatera mätdata per frame
        self.speed_data.update(state_lower, tracked, self.config.fps)

        if tracked:
            hip_x = tracked.get_center_x()
            hip_y = tracked.get_center_y()
            text = f"{state_machine.state} | x={hip_x:.2f} y={hip_y:.2f}"
            color = {
                "IDLE": (200, 200, 200),
                "MOVING": (0, 255, 0),
                "AIRBORNE": (0, 165, 255),
            }.get(state_machine.state, (255, 255, 255))
        else:
            text = f"SEARCHING"
            color = (0, 0, 255)

        # Extra rader: visa senaste speed/airtime
        speed_status = "ON" if self.speed_data.active else "OFF"
        speed_text = f"latest speed {self.speed_data.last_speed:.3f} u/s [{speed_status}]"

        last_airtime = self.airborne_data.get_last_airtime()
        last_airborne_frames = self.airborne_data.get_last_airborne_frames()
        jump_len = self.airborne_data.get_last_jump_len()
        if jump_len is None:
            jump_text = "jump len n/a"
        else:
            jump_text = f"jump len {jump_len:.1f} cm"
        airtime_text = f"latest airtime {last_airtime:.2f} s ({last_airborne_frames} frames) | {jump_text}"

        # Draw background + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        feedback_scale = 1.5
        scale = 1.2 * feedback_scale
        thickness = max(2, int(2 * feedback_scale))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        (tw_speed, th_speed), base_speed = cv2.getTextSize(speed_text, font, scale, thickness)
        (tw2, th2), baseline2 = cv2.getTextSize(airtime_text, font, scale, thickness)
        top_margin = int(20 * feedback_scale)
        row_gap = int(12 * feedback_scale)
        total_width = max(tw, tw_speed, tw2)
        total_height = th + th_speed + th2 + baseline + base_speed + baseline2 + row_gap * 3
        x = 10
        panel_top = max(0, top_margin)
        y = panel_top + th

        cv2.rectangle(
            frame,
            (x - int(8 * feedback_scale), panel_top - int(8 * feedback_scale)),
            (x + total_width + int(10 * feedback_scale), panel_top + total_height),
            (0, 0, 0),
            -1,
        )

        y1 = y
        y2 = y1 + th_speed + row_gap
        y3 = y2 + th2 + row_gap

        cv2.putText(frame, text, (x, y1), font, scale, color, thickness)
        cv2.putText(frame, speed_text, (x, y2), font, scale, (255, 255, 255), thickness)
        cv2.putText(frame, airtime_text, (x, y3), font, scale, (255, 255, 255), thickness)

        if self.length_visual is not None:
            self.length_visual.draw(frame, self.floor_tracker)

