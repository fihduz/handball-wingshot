import cv2


class MediaPlayerOverlay:
    """Visual overlay for the media player — shows state machine info etc."""

    def __init__(self, config):
        self.config = config

    def draw(self, frame, tracker):
        """Draw state machine info on the frame."""
        if tracker is None:
            return

        state_machine = tracker.state_machine
        tracked = tracker.tracked_player

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

        # Draw background + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 3.5
        thickness = 6
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = 10, 50 + th
        cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + baseline + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)
