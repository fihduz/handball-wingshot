import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

from player import Player


class MediaPipeHandler:
    """Encapsulates MediaPipe pose detection and single-player tracker."""

    def __init__(self, config):
        self.config = config
        self.pose_landmarker = self._initialize_pose_landmarker()
        self.tracked_player = None

    def _initialize_pose_landmarker(self):
        base_options = mp.tasks.BaseOptions(model_asset_path=self.config.model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=self.config.running_mode,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_pose_detection_confidence,
            min_pose_presence_confidence=self.config.min_pose_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        return PoseLandmarker.create_from_options(options)

    def detect_poses(self, frame, timestamp_ms):
        input_frame = frame
        max_width = getattr(self.config, "detection_input_max_width", None)
        if max_width and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            new_size = (max_width, int(frame.shape[0] * scale))
            input_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)

    def process_frame(self, frame, pose_result):
        if not pose_result or not pose_result.pose_landmarks:
            # No pose detected
            if self.tracked_player:
                self.tracked_player.frames_unseen += 1
                if self.tracked_player.frames_unseen > self.config.max_frames_unseen:
                    self.tracked_player = None
            return

        landmarks = pose_result.pose_landmarks[0]

        if self.tracked_player is None:
            self.tracked_player = Player(landmarks, self.config)
        else:
            # Check if pose is close enough to tracked player
            prev_x = self.tracked_player.get_center_x()
            prev_y = (self.tracked_player.landmarks[23].y + self.tracked_player.landmarks[24].y) / 2
            new_x = (landmarks[23].x + landmarks[24].x) / 2
            new_y = (landmarks[23].y + landmarks[24].y) / 2
            distance = ((prev_x - new_x) ** 2 + (prev_y - new_y) ** 2) ** 0.5

            if distance <= self.config.tracking_center_shift_threshold:
                self.tracked_player.update(landmarks, self.config)
            else:
                self.tracked_player.frames_unseen += 1
                if self.tracked_player.frames_unseen > self.config.max_frames_unseen:
                    self.tracked_player = Player(landmarks, self.config)

        if self.tracked_player:
            self._draw_skeleton(frame, self.tracked_player.landmarks)

    def get_overlay_info(self):
        if self.tracked_player:
            state_text = f"State: {self.tracked_player.state}"
        else:
            state_text = "State: SEARCHING"
        return state_text, ""

    def _draw_skeleton(self, frame, landmarks):
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


