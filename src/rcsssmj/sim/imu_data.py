from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rcsssmj.sim.sensors import IMUSensor


class ImuData:
    """IMU engine state data."""

    def __init__(self, sensors: Mapping[str, 'IMUSensor'],
                 gyro_drift_rate: float,
                 accel_drift_rate: float,
                 gyro_noise_std: float,
                 accel_noise_std: float,
                 beta: float) -> None:
        """Construct a new IMU state data.

        Parameter
        ---------
        sensors: Mapping[str, IMUSensor]
            The list of known IMU sensors.
        gyro_drift_rate, accel_drift_rate, gyro_noise_std, accel_noise_std, beta: float
            IMU simulation parameters.
        """
        self.sensors = sensors
        self.gyro_drift_rate = gyro_drift_rate
        self.accel_drift_rate = accel_drift_rate
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.beta = beta
