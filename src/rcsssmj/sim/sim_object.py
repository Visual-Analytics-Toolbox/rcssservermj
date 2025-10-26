import numpy as np

from rcsssmj.sim.sim_atom import SimAtom


class SimObject(SimAtom):
    """Abstraction of an object in a simulation."""

    def place_at(
        self,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Place the object at the specified location.

        Note: This method will also reset the object velocities and accelerations to zero.

        Parameter
        ---------
        pos: tuple[float, float, float]
            The target position.

        quat: tuple[float, float, float, float] | None, default=None
            The target orientation. If ``None``, the "identity" quaternion is used.
        """

        # set object state
        self._qpos[0:3] = pos
        self._qpos[3:7] = (1, 0, 0, 0) if quat is None else quat

        self._qvel[0:6] = np.zeros(6)
        self._qacc[0:6] = np.zeros(6)

        # set derived state information
        self._xpos[0:3] = self._qpos[0:3].astype(np.float64)
        self._xquat[0:4] = self._qpos[3:7].astype(np.float64)
        self._prev_xpos = self._xpos.astype(np.float64)
