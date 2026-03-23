import math
from typing import Any
import numpy as np

from rcsssmj.sim.imu_data import ImuData
from rcsssmj.sim.sensors import IMUSensor


def i_compile(mj_spec: Any) -> ImuData:
    """Compile specification for IMU."""
    sensors = {site.name[:-4]: IMUSensor(site.name[:-4], site.name)
               for site in mj_spec.sites if site.name.endswith('_imu')}

    def find_custom_numeric(name: str, default: float) -> float:
        for numeric in mj_spec.custom.numeric:
            if numeric.name == name:
                return numeric.data[0]
        return default

    # Parameters based on the HiPNUC HI13R4 IMU datasheet
    gyro_drift_rate = find_custom_numeric(
        'imu_gyro_drift_rate',
        (1.6 * 3.14159265359 / 180.0) / 3600.0,
    )
    accel_drift_rate = find_custom_numeric('imu_accel_drift_rate', 18e-6 * 9.81)
    gyro_noise_std = find_custom_numeric('imu_gyro_noise_std', 0.005)
    accel_noise_std = find_custom_numeric('imu_accel_noise_std', 0.05)
    beta = find_custom_numeric('imu_beta', 0.05)

    return ImuData(sensors, gyro_drift_rate, accel_drift_rate, gyro_noise_std, accel_noise_std, beta)


def i_recompile(mj_spec: Any, old_data: ImuData) -> ImuData:
    """Recompile specification preserving old drift states."""
    new_data = i_compile(mj_spec)

    for name, sensor in new_data.sensors.items():
        old_sensor = old_data.sensors.get(name, None)
        if old_sensor is not None:
            sensor.q_est[:] = old_sensor.q_est[:]
            sensor.gyro_bias[:] = old_sensor.gyro_bias[:]
            sensor.accel_bias[:] = old_sensor.accel_bias[:]

    return new_data


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def i_step(i_data: ImuData, mj_data: Any, mj_model: Any) -> None:
    """Progress the IMU simulation (Sensor Fusion & Noise Injection)."""

    dt = mj_model.opt.timestep

    # Parameters are now in i_data
    gyro_drift_rate = i_data.gyro_drift_rate
    accel_drift_rate = i_data.accel_drift_rate
    gyro_noise_std = i_data.gyro_noise_std
    accel_noise_std = i_data.accel_noise_std
    beta = i_data.beta

    for sensor in i_data.sensors.values():
        # 1. Read the raw values from MUJoCo (without noise or drift)
        try:
            true_gyro = mj_data.sensor(f"{sensor.site}_gyro").data.astype(np.float64)
            true_accel = mj_data.sensor(f"{sensor.site}_acc").data.astype(np.float64)
        except KeyError:
            continue

        # 2. Aplly drift as a random walk process
        sensor.gyro_bias += np.random.normal(0, gyro_drift_rate * dt, 3)
        sensor.accel_bias += np.random.normal(0, accel_drift_rate * dt, 3)

        # 3. Generate the final signals with noise
        sensor.noisy_gyro = true_gyro + sensor.gyro_bias + np.random.normal(0, gyro_noise_std, 3)
        sensor.noisy_accel = true_accel + sensor.accel_bias + np.random.normal(0, accel_noise_std, 3)

        # 4. Madgwick Filter to estimate orientation in quarternion
        q = sensor.q_est
        g = sensor.noisy_gyro

        q_dot = 0.5 * np.array([
            -q[1]*g[0] - q[2]*g[1] - q[3]*g[2],
            q[0]*g[0] + q[2]*g[2] - q[3]*g[1],
            q[0]*g[1] - q[1]*g[2] + q[3]*g[0],
            q[0]*g[2] + q[1]*g[1] - q[2]*g[0]
        ], dtype=np.float64)

        a = _normalize(sensor.noisy_accel)
        if np.linalg.norm(a) > 0:
            # descedent gradient to correct roll and pitch based on gravity
            f = np.array([
                2*(q[1]*q[3] - q[0]*q[2]) - a[0],
                2*(q[0]*q[1] + q[2]*q[3]) - a[1],
                2*(0.5 - q[1]**2 - q[2]**2) - a[2]
            ], dtype=np.float64)
            J = np.array([
                [-2*q[2],  2*q[3], -2*q[0],  2*q[1]],
                [2*q[1],  2*q[0],  2*q[3],  2*q[2]],
                [0,      -4*q[1], -4*q[2],  0]
            ], dtype=np.float64)

            step = _normalize(J.T @ f)
            q_dot -= beta * step

        # update the estimated orientation by integrating the quarternion derivate
        sensor.q_est += q_dot * dt
        sensor.q_est = _normalize(sensor.q_est)
