class Pose:
    """Holds pose data for a single tracked person."""

    def __init__(self, landmarks, config):
        self.landmarks = landmarks
        self.prev_landmarks = None
        self.frames_unseen = 0

    def update(self, new_landmarks, config):
        self.prev_landmarks = self.landmarks
        self.landmarks = new_landmarks
        self.frames_unseen = 0

    def get_center_x(self):
        return (self.landmarks[23].x + self.landmarks[24].x) / 2

    def get_center_y(self):
        return (self.landmarks[23].y + self.landmarks[24].y) / 2

