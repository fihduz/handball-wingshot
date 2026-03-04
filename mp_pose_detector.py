import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions


class PoseDetector:
    """Handles MediaPipe inference only. Crops, scales, detects, remaps."""

    def __init__(self, config):
        self.config = config
        self.pose_landmarker = self._create_landmarker()

    def _create_landmarker(self):
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

    def detect(self, frame, timestamp_ms):
        """Run inference on the area left of drop_zone. Returns pose result with remapped coords."""
        h, w = frame.shape[:2]

        # Crop: blind to everything right of drop zone
        crop_right = int(w * self.config.drop_zone_left)
        cropped = frame[:, :crop_right]

        # Downscale large frames
        max_width = 1920
        if cropped.shape[1] > max_width:
            scale = max_width / cropped.shape[1]
            new_size = (max_width, int(cropped.shape[0] * scale))
            cropped = cv2.resize(cropped, new_size, interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Remap x-coordinates back to full frame
        if result and result.pose_landmarks:
            scale_x = self.config.drop_zone_left
            for landmarks in result.pose_landmarks:
                for lm in landmarks:
                    lm.x *= scale_x

        return result
