from pathlib import Path
from mediapipe.tasks.python.vision import RunningMode

# -----------------------------
# Video selection (ONLY change this line)
# -----------------------------
SELECTED_VIDEO = "vid3" 

# Video registry: just paths
VIDEO_PATHS = {
    "vid3": r"D:\wingshot\DJ - 0303\V2\f4.mp4"
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

        # Video properties (set by MediaPlayer.open via update_fps)
        self.fps = 30.0
        self.frame_delay_ms = max(1, int(1000.0 / self.fps))

        # Display settings
        self.display_width = 960
        self.display_height = 540

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
        self.playback_speed = 1.0

        # Search / drop zones
        self.search_zone_right = 0.20
        self.drop_zone_left = 0.90

        # Tracking
        self.max_frames_unseen = 10
        self.tracking_center_shift_threshold = 0.15

        # Motion/state thresholds
        self.motion_threshold = 0.00001
        self.motion_key_indices = [25, 26, 27, 28]
        self.jump_threshold = 0.004
        self.jump_frame_count = 2
        self.landing_threshold = 0.003
        self.landing_frame_count = 2

        # Pose detection confidence
        self.min_pose_detection_confidence = 0.3
        self.min_pose_presence_confidence = 0.3
        self.min_tracking_confidence = 0.3

    def update_fps(self, fps: float):
        if fps and fps > 1.0:
            self.fps = float(fps)
            self.frame_delay_ms = max(1, int(1000.0 / self.fps))

    def get_delay_ms(self) -> int:
        return max(1, int(1000.0 / (self.fps * self.playback_speed)))


def main():
    import cv2
    from mediaplayer import MediaPlayer
    from mediapipe_handler import MediaPipeHandler
    from mediaplayer_overlay import MediaPlayerOverlay

    config = Config()
    media_player = MediaPlayer(config)
    mp_handler = MediaPipeHandler(config)
    overlay = MediaPlayerOverlay(config)

    # --- Area annotation step ---
    from mediaplayer_area_annotation import AreaAnnotation
    if not media_player.open():
        print(f"Error: Could not open video {config.video_path}")
        return

    # Läs första framen
    ret, frame = media_player.read_frame()
    if not ret or frame is None:
        print("Kunde inte läsa första framen för annotering.")
        media_player.close()
        return

    annotator = AreaAnnotation(config.video_path)
    window_name = "Annotera område (tryck ESC när klar)"
    # Skala annoteringsbilden till rimlig storlek
    if frame is not None:
        temp_frame = cv2.resize(frame.copy(), (config.display_width, config.display_height))
    else:
        temp_frame = None

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not annotator.done:
            print(f"Klick vid pixel ({x}, {y})")
            try:
                real_x = float(input("Ange X (meter): "))
                real_z = float(input("Ange Z (meter): "))
            except Exception:
                print("Ogiltigt värde, försök igen.")
                return
            # Justera pixelkoordinater till originalbildens skala
            if temp_frame is not None and frame is not None:
                scale_x = frame.shape[1] / config.display_width
                scale_y = frame.shape[0] / config.display_height
                px = int(x * scale_x)
                py = int(y * scale_y)
                annotator.add_point(px, py, real_x, real_z)
                annotator.draw_overlay(temp_frame)
                cv2.imshow(window_name, temp_frame)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while not annotator.is_ready():
        display_frame = temp_frame.copy() if temp_frame is not None else None
        if display_frame is not None:
            annotator.draw_overlay(display_frame)
            cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    annotator.save()
    cv2.destroyWindow(window_name)

    # --- Vanlig videoprocess startar ---
    print(f"Video: {config.fps}fps")
    print("Controls: 'p' pause, 'j'/'l' step, 'r' reset, 'n' airborne, 'm' land, 'q' quit")

    frame_interval_ms = max(1, int(1000 / config.fps))
    virtual_timestamp_ms = 0
    last_annotated_frame = None

    while media_player.is_open:
        if not media_player.is_paused or media_player.pending_frame is not None:
            ret, frame = media_player.read_frame()
            if not ret or frame is None:
                break

            annotated_frame = frame.copy() if frame is not None else None
            if annotated_frame is not None:
                mp_handler.process(annotated_frame, virtual_timestamp_ms)
                overlay.draw(annotated_frame, mp_handler.tracker)
                virtual_timestamp_ms += frame_interval_ms
                last_annotated_frame = annotated_frame
        else:
            annotated_frame = last_annotated_frame

        if annotated_frame is not None:
            media_player.display_frame(annotated_frame)

        if not media_player.handle_input():
            break

        # Handle key-triggered actions
        if getattr(media_player, '_reset_requested', False):
            mp_handler.reset_tracking()
            media_player._reset_requested = False
        if getattr(media_player, '_landing_requested', False):
            mp_handler.tracker.state_machine.mark_landing()
            media_player._landing_requested = False
        if getattr(media_player, '_airborne_requested', False):
            mp_handler.tracker.state_machine.mark_airborne()
            media_player._airborne_requested = False

    media_player.close()


if __name__ == "__main__":
    main()
