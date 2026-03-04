class AirborneData:
    """Håller koll på senaste airtime (antal frames och sekunder)."""
    def __init__(self):
        self.last_airborne_frames = 0
        self.last_airborne_seconds = 0.0
        self._airborne_active = False
        self._frame_counter = 0

    def update(self, state, fps):
        # Om vi är airborne, räkna frames
        if state == "airborne":
            if not self._airborne_active:
                self._airborne_active = True
                self._frame_counter = 1
            else:
                self._frame_counter += 1
        else:
            if self._airborne_active:
                # Vi har precis lämnat airborne
                self.last_airborne_frames = self._frame_counter
                self.last_airborne_seconds = self._frame_counter / fps if fps else 0.0
                self._airborne_active = False
                self._frame_counter = 0

    def get_last_airtime(self):
        return self.last_airborne_seconds
