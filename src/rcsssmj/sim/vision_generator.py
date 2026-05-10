import random
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
from numpy.typing import NDArray

from rcsssmj.sim.perceptions import AgentDetection, ObjectDetection, PObjectDetection, VisionPerception
from rcsssmj.sim.sensors import Camera


class VisionGenerator(ABC):
    """Abstract base class for all vision generation models."""

    @abstractmethod
    def generate(self, v_sensor: Camera, observer_prefix: str) -> VisionPerception:
        """Main orchestrator method for converting sensor data into a VisionPerception object."""


class OfficialVisionGenerator(VisionGenerator):
    """Realistic vision model simulating physical sensor limits, computer vision noise, and neural network failures."""

    def __init__(
        self,
        fov_h: float = 60.0,
        fov_v: float = 60.0,
        dist_sigma: float = 0.01,
        angle_sigma: float = 0.5,
        fn_rate: float = 0.01,
        fp_rate: float = 0.001,
        confusion_rate: float = 0.05,
        send_unique_class_names: bool = True,  # noqa: FBT001 FBT002
        decimal_position_precision: int = 2,
        max_number_of_false_positives: int = 5,
    ):
        """Initializes the OfficialVisionGenerator with configured visual constraints and statistical error parameters.

        Parameter
        ---------
        fov_h: float, default=60.0
            The horizontal field of view half-angle in degrees.

        fov_v: float, default=60.0
            The vertical field of view half-angle in degrees.

        dist_sigma: float, default=0.01
            The standard deviation for distance noise (scales proportionally with distance).

        angle_sigma: float, default=0.5
            The standard deviation for angle noise (azimuth and elevation) in degrees.

        fn_rate: float, default=0.01
            The probability of failing to detect a fully visible object (False Negative).

        fp_rate: float, default=0.001
            The probability per frame of hallucinating a non-existent object (False Positive).

        confusion_rate: float, default=0.05
            The probability of misclassifying an object (e.g., Goal as Ball).

        send_unique_class_names: bool, default=True
            If ``True``, abbreviate static world object names to a single uppercase letter.

        decimal_position_precision: int, default=2
            The decimal precision for truncating coordinates.

        max_number_of_false_positives: int, default=5
            The absolute maximum number of fake objects the camera can hallucinate per frame.
        """

        self.fov_h: float = fov_h
        """The horizontal field of view half-angle in degrees."""

        self.fov_v: float = fov_v
        """The vertical field of view half-angle in degrees."""

        self.dist_sigma: float = dist_sigma
        """The standard deviation for distance noise (scales proportionally with distance)."""

        self.angle_sigma: float = angle_sigma
        """The standard deviation for angle noise (azimuth and elevation) in degrees."""

        self.fn_rate: float = fn_rate
        """The probability of failing to detect a fully visible object (False Negative)."""

        self.fp_rate: float = fp_rate
        """The probability per frame of hallucinating a non-existent object (False Positive)."""

        self.confusion_rate: float = confusion_rate
        """The probability of misclassifying an object (e.g., Goal as Ball)."""

        self.send_unique_class_names: bool = send_unique_class_names
        """If ``True``, abbreviate static world object names to a single uppercase letter."""

        self.decimal_position_precision: int = decimal_position_precision
        """The decimal precision for truncating coordinates."""

        self.max_number_of_false_positives: int = max_number_of_false_positives
        """The absolute maximum number of fake objects the camera can hallucinate per frame."""

    def filter_fov(self, v_sensor: Camera) -> NDArray[np.bool]:
        """Filters objects against the camera's horizontal and vertical Field of View.

        Returns
        -------
        mask: NDArray[np.bool]
            A boolean mask where ``True`` indicates the object is within the FOV.
        """

        return (v_sensor.azimuths >= -self.fov_h) & (v_sensor.azimuths <= self.fov_h) & (v_sensor.elevations >= -self.fov_v) & (v_sensor.elevations <= self.fov_v)

    def apply_false_negatives(self, visibility_mask: NDArray[np.bool]) -> NDArray[np.bool]:
        """Stochastically drops objects that are otherwise visible.

        Simulates False Negatives typical of neural network object detectors (e.g., YOLO).
        """

        new_mask: NDArray[np.bool] = visibility_mask.copy()
        n_items = len(new_mask)

        # Drop truly visible objects based on fn_rate
        if self.fn_rate > 0:
            drops = np.random.random(n_items) < self.fn_rate
            new_mask[new_mask & drops] = False

        return new_mask

    def generate_false_positives(self, all_world_names: NDArray[np.str_]) -> list[ObjectDetection]:
        """Simulates network hallucinations by generating fake objects (False Positives).

        The fake objects are placed at random coordinates within the camera's FOV.
        """

        obj_detections: list[ObjectDetection] = []

        # Calculate the number of hallucinations occurring in this frame
        n_false_positives = (np.random.random(self.max_number_of_false_positives) < self.fp_rate).sum()

        # Fast exit if no hallucinations triggered
        if n_false_positives == 0:
            return obj_detections

        # Generate fake polar coordinates bounded by the FOV and a plausible distance range [0.5m, 15.0m]
        fake_azis = np.random.uniform(-self.fov_h, self.fov_h, n_false_positives)
        fake_eles = np.random.uniform(-self.fov_v, self.fov_v, n_false_positives)
        fake_dists = np.random.uniform(0.5, 15.0, n_false_positives)

        # Assign random labels to the fake objects
        fake_names = np.random.choice(all_world_names, size=n_false_positives)

        for name, azi, ele, dist in zip(fake_names, fake_azis, fake_eles, fake_dists, strict=False):
            obj_name = name[0].upper() if self.send_unique_class_names else name
            obj_detections.append(ObjectDetection(obj_name, azi, ele, dist))

        return obj_detections

    def apply_noise(
        self,
        distances: NDArray[np.float64],
        azimuths: NDArray[np.float64],
        elevations: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Applies Gaussian noise to the raw polar coordinates of visible objects, representing real-world sensor inaccuracies and estimation errors."""

        n_visible = len(distances)
        if n_visible == 0:
            return distances, azimuths, elevations

        # Distance noise standard deviation is proportional to the object's actual distance
        dist_noise = np.random.normal(0.0, distances * self.dist_sigma, n_visible)
        azi_noise = np.random.normal(0.0, self.angle_sigma, n_visible)
        ele_noise = np.random.normal(0.0, self.angle_sigma, n_visible)

        # Apply the generated noise. np.maximum bounds distances at 0 to avoid physically impossible negative distances.
        noisy_dists = np.maximum(0.0, distances + dist_noise)
        noisy_azis = azimuths + azi_noise
        noisy_eles = elevations + ele_noise

        return noisy_dists, noisy_azis, noisy_eles

    def _trunc(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Utility function to truncate decimal values to the configured precision limit."""

        factor = 10**self.decimal_position_precision
        return cast(NDArray[np.float64], np.trunc(values * factor) / factor)

    def generate(self, v_sensor: Camera, observer_prefix: str) -> VisionPerception:
        """Template method orchestrating the complete vision synthesis pipeline: FOV filtering, ground truth occlusion handling, statistical failures, noise injection, and struct formatting."""

        if v_sensor is None:
            return VisionPerception('See', [])

        # Step 1: Filter by absolute Field of View constraints
        fov_mask = self.filter_fov(v_sensor)

        # Step 2: Remove physically occluded objects and the observer's own body parts
        not_occluded = ~v_sensor.is_occluded
        visibility_mask = fov_mask & not_occluded & (v_sensor.owner_ids != observer_prefix)

        # Step 3: Inject statistical object detection failures (False Negatives)
        final_mask = self.apply_false_negatives(visibility_mask)

        obj_detections: list[PObjectDetection] = []
        n_visible = np.count_nonzero(final_mask)

        # Cache available static world object labels for hallucination/confusion logic later
        all_world_names = np.unique(v_sensor.marker_names[v_sensor.owner_ids == ''])

        # ---------------------------------------------------------------------
        # Step 4: Inject Hallucinations (False Positives)
        # ---------------------------------------------------------------------
        if self.fp_rate > 0 and len(all_world_names) > 0:
            false_positives = self.generate_false_positives(all_world_names)
            obj_detections.extend(false_positives)

        # Process genuine visible objects that successfully passed all filters
        if n_visible > 0:
            # Mask out occluded/dropped targets and keep only surviving data
            v_dists = v_sensor.distances[final_mask]
            v_azis = v_sensor.azimuths[final_mask]
            v_eles = v_sensor.elevations[final_mask]
            v_names = v_sensor.marker_names[final_mask]
            v_owners = v_sensor.owner_ids[final_mask]

            # Step 5: Perturb coordinates with Gaussian noise
            n_dists, n_azis, n_eles = self.apply_noise(v_dists, v_azis, v_eles)

            # Step 6: Truncate floats to reduce outgoing network bandwidth payload
            n_dists = self._trunc(n_dists)
            n_azis = self._trunc(n_azis)
            n_eles = self._trunc(n_eles)

            # Identify static world markers (flags, goals) versus dynamic agent body parts
            world_mask = v_owners == ''

            # ---------------------------------------------------------------------
            # Step 7a: Process Genuine World Objects (Includes Class Confusion)
            # ---------------------------------------------------------------------
            if np.any(world_mask):
                for name, azi, ele, dist in zip(v_names[world_mask], n_azis[world_mask], n_eles[world_mask], n_dists[world_mask], strict=False):
                    obj_name = name
                    if self.confusion_rate > 0 and len(all_world_names) > 1 and random.random() < self.confusion_rate:
                        wrong_names = [n for n in all_world_names if n != name]
                        obj_name = random.choice(wrong_names)

                    if self.send_unique_class_names:
                        obj_name = obj_name[0].upper()

                    obj_detections.append(ObjectDetection(obj_name, azi, ele, dist))

        # ---------------------------------------------------------------------
        # Step 7b: Shuffle detected objects
        # ---------------------------------------------------------------------
        # Prevents clients from exploiting fixed object ordering in the perception string
        if len(obj_detections) > 0:
            n_shift = np.random.randint(0, len(obj_detections))
            obj_detections = obj_detections[-n_shift:] + obj_detections[:-n_shift]

        # ---------------------------------------------------------------------
        # Step 7c: Group Player Body Parts
        # ---------------------------------------------------------------------
        if n_visible > 0:
            unique_players = np.unique(v_owners[~world_mask])
            for p_prefix in unique_players:
                p_mask = v_owners == p_prefix

                # Parse team identifier and jersey number from the MuJoCo prefix string (e.g., 'r-red-1-')
                prefix_parts = p_prefix.split('-')
                team_name = prefix_parts[1]
                player_no = int(prefix_parts[2])

                agent_parts = []
                for name, azi, ele, dist in zip(v_names[p_mask], n_azis[p_mask], n_eles[p_mask], n_dists[p_mask], strict=False):
                    agent_parts.append(ObjectDetection(name, azi, ele, dist))

                obj_detections.append(AgentDetection('P', team_name, player_no, agent_parts))

        return VisionPerception('See', obj_detections)
