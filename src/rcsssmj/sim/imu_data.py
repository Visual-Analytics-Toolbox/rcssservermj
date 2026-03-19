from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rcsssmj.sim.sensors import T1IMUSensor

class ImuData:
    """IMU engine state data."""

    def __init__(self, sensors: Mapping[str, 'T1IMUSensor']) -> None:
        """Construct a new IMU state data.

        Parameter
        ---------
        sensors: Mapping[str, IMUSensor]
            The list of known IMU sensors.
        """
        self.sensors: Mapping[str, 'T1IMUSensor'] = sensors
        """The list of known IMU sensors."""