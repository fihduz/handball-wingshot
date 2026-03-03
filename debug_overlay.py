import cv2


class DebugOverlay:
    """Handles debug visualization overlays."""

    def __init__(self, config):
        self.config = config

    def draw_tracking_info(self, frame, tracked_player):
        if not self.config.debug_mode:
            return frame

        if tracked_player:
            text = f"TRACKING: {tracked_player.state}"
            color = (0, 255, 0)
        else:
            text = "SEARCHING"
            color = (0, 165, 255)

        cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame
