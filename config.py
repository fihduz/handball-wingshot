from pathlib import Path
from mediapipe.tasks.python.vision import RunningMode

# -----------------------------
# Video selection (ONLY change this line)
# -----------------------------
SELECTED_VIDEO = "vid2"

# Video registry
VIDEO_PATHS = {
    "vid1": r"D:\0202V1 - Trim.mp4",
    "vid2": r"D:\Videoprojekt 2.mp4",
}


class Config:
    """All configuration settings for video processing and pose detection."""

    def __init__(self):
        base_dir = Path(__file__).resolve().parent

        if SELECTED_VIDEO not in VIDEO_PATHS:
            raise ValueError(f"Unknown SELECTED_VIDEO='{SELECTED_VIDEO}'")

        raw_video_path = VIDEO_PATHS[SELECTED_VIDEO]
        video_path_obj = Path(raw_video_path)
        if not video_path_obj.is_absolute():
            video_path_obj = (base_dir / video_path_obj).resolve()

        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video not found: {video_path_obj}")

        self.video_path = str(video_path_obj)
        self.model_path = str((base_dir / "pose_landmarker_heavy.task").resolve())

        # Display settings
        self.display_width = 960
        self.display_height = 540
        self.debug_mode = True

        # MediaPipe runtime
        self.running_mode = RunningMode.VIDEO

        # Pose visualization
        self.pose_connections = [
            (11, 12), (11, 23), (23, 24), (12, 24),
            (11, 13), (13, 15), (15, 17), (17, 19),
            (12, 14), (14, 16), (16, 18), (18, 20),
            (23, 25), (25, 27), (27, 29),
            (24, 26), (26, 28), (28, 30),
        ]

        # Runtime / playback
        self.fps = 25.0
        self.frame_delay_ms = 40
        self.playback_speed = 1.0

        # Tracking
        self.max_frames_unseen = 10
        self.tracking_center_shift_threshold = 0.15

        # Motion/state thresholds
        self.motion_threshold = 0.00001
        self.motion_key_indices = [25, 26, 27, 28]
        self.jump_threshold = 0.004
        self.jump_frame_count = 3
        self.landing_threshold = 0.003
        self.landing_frame_count = 2

        # Pose detection confidence
        self.min_pose_detection_confidence = 0.5
        self.min_pose_presence_confidence = 0.5
        self.min_tracking_confidence = 0.5

    def update_fps(self, fps: float):
        if fps and fps > 1.0:
            self.fps = float(fps)
            self.frame_delay_ms = max(1, int(1000.0 / self.fps))

    def get_delay_ms(self) -> int:
        return max(1, int(1000.0 / (self.fps * self.playback_speed)))


def main():
    from mediaplayer import MediaPlayer
    from mediapipe_handler import MediaPipeHandler

    config = Config()
    media_player = MediaPlayer(config)
    mp_handler = MediaPipeHandler(config)

    if not media_player.open():
        print(f"Error: Could not open video {config.video_path}")
        return

    print(f"Video opened. FPS: {config.fps}")
    print("Controls: '+'/'-' speed, 'p' pause, 'j'/'l' step, 'q' quit")

    frame_interval_ms = max(1, int(1000 / config.fps))
    virtual_timestamp_ms = 0
    last_annotated_frame = None

    while media_player.is_open:
        if not media_player.is_paused or media_player.pending_frame is not None:
            ret, frame = media_player.read_frame()
            if not ret:
                break

            pose_result = mp_handler.detect_poses(frame, virtual_timestamp_ms)
            virtual_timestamp_ms += frame_interval_ms

            annotated_frame = frame.copy()
            mp_handler.process_frame(annotated_frame, pose_result)
            state_text, _ = mp_handler.get_overlay_info()
            last_annotated_frame = annotated_frame
        else:
            annotated_frame = last_annotated_frame
            state_text, _ = mp_handler.get_overlay_info()

        if annotated_frame is not None:
            info = f"Frame: {media_player.frame_idx}"
            if media_player.is_paused:
                info += " (paused)"
            media_player.display_frame(annotated_frame, info_text=info, state_text=state_text)

        if not media_player.handle_input():
            break

    media_player.close()


if __name__ == "__main__":
    main()
