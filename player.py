import numpy as np


class Player:
    """Tracks a single person with a minimal state machine."""

    def __init__(self, landmarks, config):
        self.landmarks = landmarks
        self.prev_landmarks = None
        self.state = "STANDING"
        self.motion_score = 0.0
        self.frames_unseen = 0
        self.prev_hip_y = self._hip_center_y(landmarks)
        self.hip_rise_frames = 0
        self.fall_frames = 0

    def update(self, new_landmarks, config):
        self.prev_landmarks = self.landmarks
        self.landmarks = new_landmarks
        self.frames_unseen = 0
        self._update_state(config)

    def get_center_x(self):
        return (self.landmarks[23].x + self.landmarks[24].x) / 2

    def _update_state(self, config):
        if self.prev_landmarks is None:
            return

        self.motion_score = self._compute_motion(config)
        hip_y = self._hip_center_y(self.landmarks)
        hip_delta = self.prev_hip_y - hip_y if self.prev_hip_y is not None else 0

        if hip_delta > config.jump_threshold:
            self.hip_rise_frames += 1
        else:
            self.hip_rise_frames = 0

        if hip_delta < -config.landing_threshold:
            self.fall_frames += 1
        else:
            self.fall_frames = 0

        base = "RUNNING" if self.motion_score >= config.motion_threshold else "STANDING"

        if self.state == "JUMPING":
            if self.fall_frames >= config.landing_frame_count:
                self.state = "LANDING"
            # else stay JUMPING
        elif self.state == "LANDING":
            if self.fall_frames < config.landing_frame_count:
                self.state = base
        elif base == "RUNNING" and self.hip_rise_frames >= config.jump_frame_count:
            self.state = "JUMPING"
        else:
            self.state = base

        self.prev_hip_y = hip_y

    def _compute_motion(self, config):
        if self.prev_landmarks is None:
            return 0.0
        score = 0.0
        for idx in config.motion_key_indices:
            dx = self.landmarks[idx].x - self.prev_landmarks[idx].x
            dy = self.landmarks[idx].y - self.prev_landmarks[idx].y
            score += np.sqrt(dx * dx + dy * dy)
        return score

    def _hip_center_y(self, landmarks):
        return (landmarks[23].y + landmarks[24].y) / 2

