import math


class AirborneData:
    """Håller koll på senaste airtime (antal frames och sekunder)."""

    def __init__(self):
        self.last_airborne_frames = 0
        self.last_airborne_seconds = 0.0
        self.last_jump_len = None
        self.last_jumping_point_cm = None
        self.last_jump_valid = False
        self.history = []
        self.active = False
        self._start_frame = None
        self._start_world_foot = None

    def toggle_by_frame(self, frame_index, fps, world_foot=None, reference_point=None):
        if frame_index is None:
            return None

        if not self.active:
            self.active = True
            self._start_frame = int(frame_index)
            self._start_world_foot = world_foot
            return None

        stop_frame = int(frame_index)
        if self._start_frame is None:
            self.active = False
            return None
        frames = max(0, stop_frame - self._start_frame)
        self.last_airborne_frames = frames
        self.last_airborne_seconds = (frames / fps) if fps else 0.0

        if self._start_world_foot is not None and world_foot is not None:
            dr = float(world_foot[0]) - float(self._start_world_foot[0])
            dy = float(world_foot[1]) - float(self._start_world_foot[1])
            self.last_jump_len = math.hypot(dr, dy)
            self.last_jump_valid = True
        else:
            self.last_jump_len = None
            self.last_jump_valid = False

        # jumping_point_cm = distance between first press (takeoff foot) and point 1a.
        if self._start_world_foot is not None and reference_point is not None:
            tr = float(self._start_world_foot[0]) - float(reference_point[0])
            ty = float(self._start_world_foot[1]) - float(reference_point[1])
            self.last_jumping_point_cm = math.hypot(tr, ty)
        else:
            self.last_jumping_point_cm = None

        self.active = False
        self._start_frame = None
        self._start_world_foot = None

        record = {
            "frames": self.last_airborne_frames,
            "airtime_s": self.last_airborne_seconds,
            "jump_len_cm": self.last_jump_len,
            "jumping_point_cm": self.last_jumping_point_cm,
        }
        self.history.append(record)
        return record

    def get_last_airtime(self):
        return self.last_airborne_seconds

    def get_last_airborne_frames(self):
        return self.last_airborne_frames

    def get_last_jump_len(self):
        return self.last_jump_len

    def get_last_jumping_point_cm(self):
        return self.last_jumping_point_cm

    def get_last_takeoff_cm(self):
        # Backward-compatible alias while callers migrate to jumping_point naming.
        return self.last_jumping_point_cm


class SpeedData:
    """Mäter spelarens hastighet i MOVING-fasen mellan toggle 1 start/stopp."""

    def __init__(self):
        self.active = False
        self.last_speed = 0.0
        self._sum_speed = 0.0
        self._sample_count = 0
        self._prev_center = None

    def toggle(self):
        if not self.active:
            self.active = True
            self._sum_speed = 0.0
            self._sample_count = 0
            self._prev_center = None
            return

        self.active = False
        if self._sample_count > 0:
            self.last_speed = self._sum_speed / self._sample_count
        self._prev_center = None

    def update(self, state, tracked_pose, fps):
        if not self.active:
            return

        if tracked_pose is None or state != "moving":
            self._prev_center = None
            return

        curr_center = (tracked_pose.get_center_x(), tracked_pose.get_center_y())
        if self._prev_center is not None and fps:
            dx = curr_center[0] - self._prev_center[0]
            dy = curr_center[1] - self._prev_center[1]
            distance = math.hypot(dx, dy)
            instant_speed = distance * fps
            self._sum_speed += instant_speed
            self._sample_count += 1

        self._prev_center = curr_center


class ArmAngleData:
    """Fångar axel- och armbågsvinkel i två manuella tillfällen via toggle 2."""

    def __init__(self):
        self.active = False
        self._first_capture = None
        self.latest_first = None
        self.latest_second = None

    @staticmethod
    def _angle_degrees(a, b, c):
        bax = a[0] - b[0]
        bay = a[1] - b[1]
        bcx = c[0] - b[0]
        bcy = c[1] - b[1]

        norm_ba = math.hypot(bax, bay)
        norm_bc = math.hypot(bcx, bcy)
        if norm_ba == 0 or norm_bc == 0:
            return 0.0

        cos_theta = (bax * bcx + bay * bcy) / (norm_ba * norm_bc)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))

    def _capture(self, tracked_pose):
        if tracked_pose is None or tracked_pose.landmarks is None:
            return None

        lm = tracked_pose.landmarks
        right_hip = (lm[24].x, lm[24].y)
        right_shoulder = (lm[12].x, lm[12].y)
        right_elbow = (lm[14].x, lm[14].y)
        right_wrist = (lm[16].x, lm[16].y)

        shoulder_angle = self._angle_degrees(right_hip, right_shoulder, right_elbow)
        elbow_angle = self._angle_degrees(right_shoulder, right_elbow, right_wrist)
        return (shoulder_angle, elbow_angle)

    def toggle(self, tracked_pose):
        if not self.active:
            self._first_capture = self._capture(tracked_pose)
            self.active = True
            return

        second_capture = self._capture(tracked_pose)
        if self._first_capture is not None and second_capture is not None:
            self.latest_first = self._first_capture
            self.latest_second = second_capture
        self._first_capture = None
        self.active = False

