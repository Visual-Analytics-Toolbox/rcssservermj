from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from rcsssmj.sim.sensors import Camera


class VisionData:
    """Encapsulates pre-allocated memory buffers and metadata required by the vision pipeline."""

    def __init__(
        self,
        sensors: Mapping[str, Camera],
        marker_sites: Sequence[str],
        marker_names: NDArray[np.str_],
        owner_ids: NDArray[np.str_],
        n_world_markers: int,
        *,
        check_occlusion: bool = True,
    ) -> None:
        """Construct a new vision state data.

        Parameter
        ---------
        sensors: Mapping[str, Camera]
            The map of known camera sensors.

        marker_sites: Sequence[str]
            The list of known marker sites.

        marker_names: NDArray[np.str_]
            The list of known marker names.

        owner_ids: NDArray[np.int_]
            The list of known marker owners.

        n_world_markers: int
            The number of known world markers.

        check_occlusion: bool, default=False
            Whether to check for occlusions in the vision engine.
            ``True`` to account for object occlusions, ``False`` for X-Ray vision.
        """

        self.sensors: Mapping[str, Camera] = sensors
        """The map of known camera sensors."""

        self.marker_sites: Sequence[str] = marker_sites
        """The list of known marker sites."""

        self.marker_names: NDArray[np.str_] = marker_names
        """The list of known marker names."""

        self.owner_ids: NDArray[np.str_] = owner_ids
        """The list of known marker owners."""

        self.n_world_markers: int = n_world_markers
        """The number of known world markers."""

        self.check_occlusion: bool = check_occlusion
        """Whether to check for occlusions in the vision engine."""

        n_markers = len(marker_sites)
        self.obj_pos: NDArray[np.float64] = np.zeros((3, n_markers), dtype=np.float64)
        """The array of known marker positions."""
