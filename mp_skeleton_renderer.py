import cv2


class SkeletonRenderer:
    """Draws pose skeleton on a frame."""

    def __init__(self, config):
        self.config = config

    def draw(self, frame, landmarks):
        """Draw skeleton connections and joint dots for one set of landmarks."""
        if landmarks is None:
            return
        h, w = frame.shape[:2]

        for start_idx, end_idx in self.config.pose_connections:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            if landmarks[start_idx].visibility < 0.5 or landmarks[end_idx].visibility < 0.5:
                continue
            x0, y0 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
            x1, y1 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
            cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        for lm in landmarks:
            if lm.visibility < 0.5:
                continue
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
