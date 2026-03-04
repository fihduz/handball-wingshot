class StateMachine:
    """
    Player state: IDLE -> MOVING -> AIRBORNE.

    Transitions:
        IDLE -> MOVING:     auto (feet move in X for 3 frames AND hips shift for 2 frames)
        MOVING -> AIRBORNE: manual (user presses 'n')
        AIRBORNE -> MOVING: manual (user presses 'm')

    Thresholds scale with torso height to handle perspective changes.
    """

    IDLE = "IDLE"
    MOVING = "MOVING"
    AIRBORNE = "AIRBORNE"

    def __init__(self, config):
        self.config = config
        self.state = self.IDLE
        self.feet_moving_frames = 0
        self.hip_moving_frames = 0

    # ---- helpers ----

    def _torso_height(self, landmarks):
        """Mid-shoulder to mid-hip distance — body-scale reference."""
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        return abs(hip_y - shoulder_y)

    @staticmethod
    def _mid_foot_x(landmarks):
        return (landmarks[27].x + landmarks[28].x) / 2

    @staticmethod
    def _mid_hip_x(landmarks):
        return (landmarks[23].x + landmarks[24].x) / 2

    @staticmethod
    def _mid_hip_y(landmarks):
        return (landmarks[23].y + landmarks[24].y) / 2

    # ---- main update ----

    def update(self, pose):
        """Call once per frame. Only auto-detects IDLE -> MOVING."""
        if pose.landmarks is None or pose.prev_landmarks is None:
            return

        # Only auto-detect in IDLE state
        if self.state != self.IDLE:
            return

        curr = pose.landmarks
        prev = pose.prev_landmarks
        torso_h = self._torso_height(curr)

        feet_dx = abs(self._mid_foot_x(curr) - self._mid_foot_x(prev))
        hip_dx = abs(self._mid_hip_x(curr) - self._mid_hip_x(prev))
        hip_dy = abs(self._mid_hip_y(curr) - self._mid_hip_y(prev))

        feet_move_thresh = torso_h * 0.02
        hip_move_thresh = torso_h * 0.01

        if feet_dx > feet_move_thresh:
            self.feet_moving_frames += 1
        else:
            self.feet_moving_frames = 0

        if hip_dx > hip_move_thresh or hip_dy > hip_move_thresh:
            self.hip_moving_frames += 1
        else:
            self.hip_moving_frames = 0

        if self.feet_moving_frames >= 3 and self.hip_moving_frames >= 2:
            self.state = self.MOVING
            self.feet_moving_frames = 0
            self.hip_moving_frames = 0

    # ---- manual transitions ----

    def mark_airborne(self):
        """Called when user presses 'n' — MOVING -> AIRBORNE."""
        if self.state == self.MOVING:
            self.state = self.AIRBORNE

    def mark_landing(self):
        """Called when user presses 'm' — AIRBORNE -> MOVING."""
        if self.state == self.AIRBORNE:
            self.state = self.MOVING

    def reset(self):
        self.state = self.IDLE
        self.feet_moving_frames = 0
        self.hip_moving_frames = 0
