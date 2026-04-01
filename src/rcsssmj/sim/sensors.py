from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class CustomSensor:
    """Base class for custom sensors."""

    def __init__(self, name: str, site: str):
        """Construct a new custom sensor representation.

        Parameter
        ---------
        name: str
            The name of the sensor.

        site: str
            The mujoco site name associated with this sensor.
        """

        self.name: Final[str] = name
        """The name of the sensor."""

        self.site: Final[str] = site
        """The mujoco site name associated with this custom sensor."""


class Microphone(CustomSensor):
    """A microphone sensor."""

    def __init__(self, name: str, site: str) -> None:
        """Construct a new microphone sensor.

        Parameter
        ---------
        name: str
            The name of the sensor.

        site: str
            The mujoco site name associated with this sensor.
        """

        super().__init__(name, site)

        self.messages: Sequence[bytes | bytearray] = []
        """The messages received by this sensor."""

        self.sources: Sequence[str] = []
        """The message sources (actuator names)."""

        self.volumes: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        """The volumes at which the individual messages arrived."""

        self.origins: NDArray[np.float64] = np.zeros((3, 0), dtype=np.float64)
        """The sensor local origins of the sound transmissions."""

        self.distances: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        """The distances to the audio origins."""


class Camera(CustomSensor):
    """A vision camera sensor."""

    def __init__(self, name: str, site: str) -> None:
        super().__init__(name, site)

        self.marker_names: NDArray[np.str_] = np.zeros(0, dtype=np.str_)
        """The names of the visible markers/objects."""

        self.distances: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        """The ground-truth distances to the objects."""

        self.azimuths: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        """The ground-truth azimuth angles to the objects."""

        self.elevations: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        """The ground-truth elevation angles to the objects."""

        self.is_occluded: NDArray[np.bool_] = np.zeros(0, dtype=np.bool_)
        """Flags indicating whether the object is occluded from this camera's perspective."""

        self.owner_ids: NDArray[np.int_] = np.zeros(0, dtype=np.int_)
        """IDs indicating if the object belongs to the world (-1) or to a specific agent."""
