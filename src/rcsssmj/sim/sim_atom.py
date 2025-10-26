from typing import Any, Final, List

import numpy as np
from numpy.typing import NDArray

from rcsssmj.sim.perceptions import Perception


class SimAtom:
    """Abstraction of an atomic object in a simulation."""

    def __init__(self, name: str) -> None:
        """Construct a new simulation atom.

        Parameter
        ---------
        name: str
            The unique name of the object.
        """

        self.name: Final[str] = name
        """The unique name of the object."""

        self._xpos: NDArray[np.float64] = np.zeros(3)
        """The root body position array view."""

        self._xquat: NDArray[np.float64] = np.array([1, 0, 0, 0])
        """The root body orientation quaternion array view."""

        self._xmat: NDArray[np.float64] = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        """The root body orientation matrix array view."""

        self._qpos: NDArray[np.float64] = np.zeros(7)
        """The root body joint position and orientation quaternion array view."""

        self._qvel: NDArray[np.float64] = np.zeros(6)
        """The root body joint (linear and angular) velocity array view."""

        self._qacc: NDArray[np.float64] = np.zeros(6)
        """The root body joint (linear and angular) acceleration array view."""

        self._prev_xpos: NDArray[np.float64] = np.zeros(3)
        """The previous position of the object."""

    @property
    def xpos(self) -> NDArray[np.float64]:
        """The root body position array view."""

        return self._xpos

    @property
    def xquat(self) -> NDArray[np.float64]:
        """The root body orientation quaternion array view."""

        return self._xquat

    @property
    def xmat(self) -> NDArray[np.float64]:
        """The root body orientation matrix array view."""

        return self._xmat

    @property
    def prev_xpos(self) -> NDArray[np.float64]:
        """The previous position of the object."""

        return self._prev_xpos

    @property
    def root_body_name(self) -> str:
        """The name of the root body of the object."""

        return self.name

    @property
    def root_joint_name(self) -> str:
        """The name of the root joint of the object."""

        return self.name + '-root'

    def bind(self, mj_model: Any, mj_data: Any) -> None:
        """Bind the object to the given model and data."""

        root_body = mj_data.body(self.root_body_name)
        self._xpos = root_body.xpos
        self._xquat = root_body.xquat
        self._xmat = root_body.xmat

        root_joint = mj_data.joint(self.root_joint_name)
        self._qpos = root_joint.qpos
        self._qvel = root_joint.qvel
        self._qacc = root_joint.qacc

    def pre_step(self, mj_model: Any, mj_data: Any) -> None:
        """Method triggered before a simulation step."""

        self._prev_xpos = self._xpos.astype(np.float64)

    def post_step(self, mj_model: Any, mj_data: Any) -> None:
        """Method triggered after a simulation step."""

    def generate_perceptions(self, agent_perceptions: List[Perception]) -> None:
        """Generates perceptions for agent."""

class SimSite(SimAtom):
    """SimAtom adjusted for binding to a site."""

    def bind(self, mj_model: Any, mj_data: Any) -> None:
        site = mj_data.site(self.root_body_name)
        self._xpos = site.xpos
        self._xmat = site.xmat
