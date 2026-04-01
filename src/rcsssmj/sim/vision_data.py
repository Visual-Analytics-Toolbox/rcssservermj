from collections.abc import Mapping

import numpy as np

from rcsssmj.sim.sensors import Camera


class VisionData:
    """Encapsulates pre-allocated memory buffers and metadata required by the vision pipeline."""

    def __init__(self, sensors: Mapping[str, Camera], marker_sites: list[str], marker_names: np.ndarray, owner_ids: np.ndarray, n_world_markers: int) -> None:
        self.sensors = sensors
        self.marker_sites = marker_sites
        self.marker_names = marker_names
        self.owner_ids = owner_ids
        self.n_world_markers = n_world_markers

        n_markers = len(marker_sites)
        self.obj_pos: np.ndarray = np.zeros((3, n_markers), dtype=np.float64)
