from p_pose import Pose
from p_statemachine import StateMachine


class PlayerTracker:
    """Manages search/track lifecycle for a single player."""

    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"

    def __init__(self, config):
        self.config = config
        self.tracked_player = None
        self.state_machine = StateMachine(config)
        self.state = self.SEARCHING

    def reset(self):
        """Drop current player and return to searching."""
        self.tracked_player = None
        self.state_machine.reset()
        self.state = self.SEARCHING

    def update(self, pose_result):
        """Process a pose result. Returns the tracked Pose or None."""
        if self.state == self.SEARCHING:
            self._search(pose_result)
        elif self.state == self.TRACKING:
            self._track(pose_result)

        # Update state machine if we have a tracked player
        if self.tracked_player:
            self.state_machine.update(self.tracked_player)

        return self.tracked_player

    def _search(self, pose_result):
        """Look for a new player in the search zone."""
        if not pose_result or not pose_result.pose_landmarks:
            return

        for landmarks in pose_result.pose_landmarks:
            hip_x = (landmarks[23].x + landmarks[24].x) / 2
            if hip_x <= self.config.search_zone_right:
                self.tracked_player = Pose(landmarks, self.config)
                self.state = self.TRACKING
                return

    def _track(self, pose_result):
        """Follow the tracked player. Drop if lost or past drop zone."""
        if not pose_result or not pose_result.pose_landmarks:
            if self.tracked_player:
                self.tracked_player.frames_unseen += 1
                if self.tracked_player.frames_unseen > self.config.max_frames_unseen:
                    self.reset()
            return

        # Find closest pose to tracked player
        best_landmarks = None
        best_distance = float("inf")
        prev_x = self.tracked_player.get_center_x()
        prev_y = (self.tracked_player.landmarks[23].y + self.tracked_player.landmarks[24].y) / 2

        for landmarks in pose_result.pose_landmarks:
            new_x = (landmarks[23].x + landmarks[24].x) / 2
            new_y = (landmarks[23].y + landmarks[24].y) / 2
            distance = ((prev_x - new_x) ** 2 + (prev_y - new_y) ** 2) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best_landmarks = landmarks

        if best_landmarks and best_distance <= self.config.tracking_center_shift_threshold:
            self.tracked_player.update(best_landmarks, self.config)
            self.tracked_player.frames_unseen = 0

            # Auto-drop if player passed drop zone
            if self.tracked_player.get_center_x() >= self.config.drop_zone_left:
                self.reset()
        else:
            self.tracked_player.frames_unseen += 1
            if self.tracked_player.frames_unseen > self.config.max_frames_unseen:
                self.reset()
