
import cv2
from p_data import AirborneData


class MediaPlayerOverlay:
    """Visual overlay for the media player — shows state machine info etc."""

    def __init__(self, config):
        self.config = config
        self.airborne_data = AirborneData()

    def draw(self, frame, tracker):
        """Draw state machine info on the frame."""
        if tracker is None:
            return

        state_machine = tracker.state_machine
        tracked = tracker.tracked_player

        # Uppdatera airborne-data
        self.airborne_data.update(state_machine.state.lower(), self.config.fps)

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

        # Extra rad: visa senaste airtime
        last_airtime = self.airborne_data.get_last_airtime()
        airtime_text = f"last airtime {last_airtime:.2f} s"

        # Draw background + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 3.5
        thickness = 6
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        (tw2, th2), baseline2 = cv2.getTextSize(airtime_text, font, scale, thickness)
        x, y = 10, 50 + th
        # Svart bakgrund för båda rader
        total_height = th + th2 + baseline + baseline2 + 20
        cv2.rectangle(frame, (x - 5, y - th - 5), (x + max(tw, tw2) + 5, y + th2 + baseline2 + 15), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)
        cv2.putText(frame, airtime_text, (x, y + th2 + 10), font, scale, (255,255,255), thickness)
