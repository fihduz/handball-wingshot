import cv2
from debug_overlay import DebugOverlay


class MediaPlayer:
    """Manages video playback and keyboard controls."""

    def __init__(self, config):
        self.config = config
        self.cap = None
        self.frame_idx = 0
        self.is_open = False
        self.is_paused = False
        self.last_frame = None
        self.pending_frame = None
        self.frame_count = None
        self.current_frame_index = -1
        self.debug_overlay = DebugOverlay(config)

    def open(self):
        self.cap = cv2.VideoCapture(self.config.video_path)
        self.is_open = self.cap.isOpened()
        if self.is_open:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.config.update_fps(fps)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        return self.is_open

    def read_frame(self):
        if not self.is_open:
            return False, None

        if self.pending_frame is not None:
            frame = self.pending_frame
            self.pending_frame = None
            self.last_frame = frame
            self.current_frame_index = self.frame_idx - 1
            return True, frame

        ret, frame = self.cap.read()
        if ret:
            self.frame_idx += 1
            self.last_frame = frame
            self.current_frame_index = self.frame_idx - 1
        return ret, frame

    def display_frame(self, frame, info_text="", state_text=""):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (5, 5), (400, 80), (0, 0, 0), -1)

        speed_text = f"Speed: {self.config.playback_speed:.1f}x"
        cv2.putText(frame, speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if info_text:
            cv2.putText(frame, info_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if state_text:
            cv2.putText(frame, state_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        display = cv2.resize(frame, (self.config.display_width, self.config.display_height))
        cv2.imshow("PoseLandmarker", display)

    def handle_input(self):
        key = cv2.waitKey(self.config.get_delay_ms()) & 0xFF

        if key == ord("q"):
            return False
        elif key in (ord("+"), ord("=")):
            self.config.playback_speed = min(3.0, self.config.playback_speed + 0.1)
        elif key == ord("-"):
            self.config.playback_speed = max(0.1, self.config.playback_speed - 0.1)
        elif key == ord("p"):
            self.is_paused = not self.is_paused
        elif key == ord("j"):
            self._seek(-1)
        elif key == ord("l"):
            self._seek(1)
        return True

    def _seek(self, delta):
        if not self.is_open or not self.cap:
            return
        target = max(0, self.current_frame_index + delta)
        if self.frame_count:
            target = min(self.frame_count - 1, target)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = self.cap.read()
        if ret:
            self.frame_idx = target + 1
            self.current_frame_index = target
            self.last_frame = frame
            self.pending_frame = frame
            self.is_paused = True

    def close(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
