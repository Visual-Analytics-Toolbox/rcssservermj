import os
from importlib import resources
from typing import TYPE_CHECKING, Any

import mujoco

if TYPE_CHECKING:
    from importlib.abc import Traversable


class ModelSpecProvider:
    """Simple class for loading robot and environment specs."""

    def load_robot_spec(self, robot: str, variant: str | None = None) -> Any | None:
        """Load the robot model specification for the given robot name and variant.

        Parameter
        ---------
        robot: str
            The name of the robot model.

        variant: str | None, default=None
            The name of the model variant. If `None` the default variant is used.
        """

        # load model
        return self.load_model_file('robots', robot, 'robot.xml' if variant is None else variant + '.xml')

    def load_environment_spec(self, environment: str, obj: str) -> Any | None:
        """Load the object model specification for the given environment and object name.

        Parameter
        ---------
        environment: str
            The name of the environment.

        obj: str | None
            The name of the environment object to load.
        """

        return self.load_model_file('environments', environment, obj + '.xml')

    def load_model_file(self, section: str, sub_section: str, name: str) -> Any | None:
        """Load the model with the given name from the resource section.

        Parameter
        ---------
        section: str
            The resource section.

        sub_section: str
            The resource sub-section.

        name: str
            The name of the model file.
        """

        # fetch robot model dir
        model_dir = os.path.join(os.path.dirname(__file__), section, sub_section)
        if not os.path.isdir(model_dir):
            return None

        # fetch model file path
        model_file = os.path.join(model_dir, name)
        if not os.path.isfile(model_file):
            return None

        # create new mujoco spec
        return mujoco.MjSpec.from_file(model_file)

    def load_model_resource(self, section: str, sub_section: str, name: str) -> Any | None:
        """Load the model with the given name from the resource section.

        Parameter
        ---------
        section: str
            The resource section.

        sub_section: str
            The resource sub-section.

        name: str
            The name of the model file.
        """

        # fetch package traversable
        package_dir: Traversable = resources.files('rcsssmj')

        # fetch robot model dir
        model_dir: Traversable = package_dir.joinpath('resources').joinpath(section).joinpath(sub_section)
        if not model_dir.is_dir():
            return None

        # fetch model file
        model_file: Traversable = model_dir.joinpath(name)
        if not model_file.is_file():
            return None

        # load model xml
        model_xml = model_file.read_text('UTF-8')

        # load model assets
        assets: dict[str, bytes] = {}

        for res in model_dir.iterdir():
            if res.is_file() and res.name != name:
                assets[res.name] = res.read_bytes()

        # create new mujoco spec
        return mujoco.MjSpec.from_string(model_xml, assets=assets)
