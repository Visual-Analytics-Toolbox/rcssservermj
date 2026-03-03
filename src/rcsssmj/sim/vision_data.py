from collections.abc import Mapping
from rcsssmj.sim.sensors import Camera

class VisionData:
    """Vision engine state data."""

    def __init__(self, sensors: Mapping[str, Camera]) -> None:
        self.sensors: Mapping[str, Camera] = sensors
        """The list of known camera sensors."""