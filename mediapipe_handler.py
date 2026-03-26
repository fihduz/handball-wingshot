from mp_pose_detector import PoseDetector
from mp_player_tracker import PlayerTracker
from mp_skeleton_renderer import SkeletonRenderer


class MediaPipeHandler:
    """Facade: ties together detection, tracking and rendering."""

    def __init__(self, config):
        self.detector = PoseDetector(config)
        self.tracker = PlayerTracker(config)
        self.renderer = SkeletonRenderer(config)

    def process(self, frame, timestamp_ms, draw_skeleton=False):
        """Run full pipeline: detect → track, and optionally draw skeleton on the frame."""
        pose_result = self.detector.detect(frame, timestamp_ms)
        player = self.tracker.update(pose_result)
        if draw_skeleton and player:
            self.renderer.draw(frame, player.landmarks)

    def reset_tracking(self):
        self.tracker.reset()

    @property
    def tracked_player(self):
        return self.tracker.tracked_player


