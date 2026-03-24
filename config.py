from pathlib import Path
from mediapipe.tasks.python.vision import RunningMode

# -----------------------------
# Video selection (ONLY change this line)
# -----------------------------
SELECTED_VIDEO = "herf2"  # options: "app12", "djf3", "fap12", "p14f2", "herf1"

# Video registry: just paths
VIDEO_PATHS = {
    "app12": r"D:\wingshot\PA - 0223\V2\p11.mp4", #fungerar inte men 0.64 s airborne, precision i avstamp 
    "djf3": r"D:\wingshot\DJ - 0303\V1\f1.mp4", #works
    "fap12": r"D:\wingshot\FA - 0223\V2\p12.mp4", #works
    "p14f2": r"D:\wingshot\P14 - 0310\V2\p11.mp4",
    "herf2": r"D:\wingshot\HER - 0317\V2\f2.mp4",
    "herf1": r"D:\wingshot\HER - 0317\V2\f1.mp4",
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
        self.video_id = SELECTED_VIDEO
        self.angle_id = "v2"
        self.calibration_store_path = str((base_dir / "calibration_store.json").resolve())

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
        self.drop_zone_left = 0.85

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
        self.min_pose_detection_confidence = 0.4
        self.min_pose_presence_confidence = 0.4
        self.min_tracking_confidence = 0.4

        # Feature flags (1/0)
        self.enable_length_visual = 1

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
    from roomtracking_floor import RoomTrackingFloor
    from length_visual import LengthVisual

    config = Config()
    media_player = MediaPlayer(config)
    mp_handler = MediaPipeHandler(config)
    overlay = MediaPlayerOverlay(config)
    roomtracking_floor = RoomTrackingFloor(config.video_path)
    overlay.set_floor_tracker(roomtracking_floor)
    length_visual = LengthVisual(config, enabled=bool(config.enable_length_visual))
    overlay.set_length_visual(length_visual)

    calibration_target = len(roomtracking_floor.REAL_WORLD_POINTS)
    calibration_order = ", ".join(roomtracking_floor.POINT_LABELS)

    # Kontrollera att videon öppnas
    if not media_player.open():
        print(f"Error: Could not open video {config.video_path}")
        return
    else:
        print("Video öppnad OK!")

    # Kalibrerings-UI: klicka alla kalibreringspunkter när videon är pausad
    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not media_player.is_paused:
            return
        if roomtracking_floor.is_ready():
            return
        if last_annotated_frame is None:
            return

        frame_h, frame_w = last_annotated_frame.shape[:2]
        px_x = (x / config.display_width) * frame_w
        px_y = (y / config.display_height) * frame_h
        ok, msg = roomtracking_floor.add_pixel_point(px_x, px_y)
        next_label = roomtracking_floor.next_point_label()
        if next_label is None:
            print(f"{msg}")
        else:
            print(f"{msg}. Next: {next_label}")

    cv2.namedWindow("PoseLandmarker")
    cv2.setMouseCallback("PoseLandmarker", mouse_callback)

    # --- Vanlig videoprocess startar ---
    print(f"Video: {config.fps}fps")
    print("Controls: 'p' pause/play, 'j'/'l' step, 'r' reset, '1' speed toggle, '3' airborne toggle, 'u' undo calib point, 'c' reset calibration, 'q' quit")
    if roomtracking_floor.is_ready():
        print("Floor calibration loaded from file.")
    else:
        print(f"Pause with 'p' and click {calibration_target} floor points in order: {calibration_order}.")

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

                # Rita kalibreringshjälp på pausad bild tills kalibrering är klar
                if not roomtracking_floor.is_ready():
                    label = roomtracking_floor.next_point_label()
                    progress = roomtracking_floor.point_count()
                    info_line = f"Calibration {progress}/{calibration_target} - click: {label}"
                    cv2.putText(
                        annotated_frame,
                        info_line,
                        (20, annotated_frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )

                    for i, (px_x, px_y) in enumerate(roomtracking_floor.get_pixel_points()):
                        cx = int(round(px_x))
                        cy = int(round(px_y))
                        cv2.circle(annotated_frame, (cx, cy), 6, (255, 255, 0), -1)
                        cv2.putText(
                            annotated_frame,
                            roomtracking_floor.POINT_LABELS[i],
                            (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )

                virtual_timestamp_ms += frame_interval_ms
                last_annotated_frame = annotated_frame
        else:
            annotated_frame = last_annotated_frame

            if annotated_frame is not None and not roomtracking_floor.is_ready():
                label = roomtracking_floor.next_point_label()
                progress = roomtracking_floor.point_count()
                info_line = f"Calibration {progress}/{calibration_target} - click: {label}"
                cv2.putText(
                    annotated_frame,
                    info_line,
                    (20, annotated_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

                for i, (px_x, px_y) in enumerate(roomtracking_floor.get_pixel_points()):
                    cx = int(round(px_x))
                    cy = int(round(px_y))
                    cv2.circle(annotated_frame, (cx, cy), 6, (255, 255, 0), -1)
                    cv2.putText(
                        annotated_frame,
                        roomtracking_floor.POINT_LABELS[i],
                        (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

        if annotated_frame is not None:
            media_player.display_frame(annotated_frame)

        was_paused = media_player.is_paused
        if not media_player.handle_input():
            break
        pause_toggled = media_player.is_paused != was_paused
        if pause_toggled and media_player.is_paused and not roomtracking_floor.is_ready():
            next_label = roomtracking_floor.next_point_label()
            print(f"Calibration mode active. Click point: {next_label}")

        # Handle key-triggered actions
        if getattr(media_player, '_reset_requested', False):
            mp_handler.reset_tracking()
            media_player._reset_requested = False
        if getattr(media_player, '_speed_toggle_requested', False):
            overlay.toggle_speed()
            media_player._speed_toggle_requested = False
        if getattr(media_player, '_airborne_toggle_requested', False):
            overlay.toggle_airborne_timer(media_player.current_frame_index, mp_handler.tracker.tracked_player)
            mp_handler.tracker.state_machine.toggle_airborne()
            media_player._airborne_toggle_requested = False
        if getattr(media_player, '_calibration_undo_requested', False):
            ok, msg = roomtracking_floor.undo_last_point()
            print(msg)
            media_player._calibration_undo_requested = False
        if getattr(media_player, '_calibration_reset_requested', False):
            roomtracking_floor.reset()
            print(f"Calibration reset. Pause and click {calibration_target} points again.")
            media_player._calibration_reset_requested = False

    media_player.close()


if __name__ == "__main__":
    main()
