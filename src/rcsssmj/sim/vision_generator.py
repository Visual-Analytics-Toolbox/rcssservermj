import random
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np

from rcsssmj.sim.perceptions import AgentDetection, ObjectDetection, PObjectDetection, VisionPerception
from rcsssmj.sim.sensors import Camera


class VisionGenerator(ABC):
    """Abstract base class for all vision generators."""

    @abstractmethod
    def generate(self, v_sensor: Camera, agent: Any, agents_list: list) -> VisionPerception:
        """Main method orchestrating the perception generation."""


class OfficialVisionGenerator(VisionGenerator):
    """Realistic implementation of vision rules, including neural network failures and noise."""

    def __init__(
        self,
        fov_h: float = 60.0,  # horizontal field of view half-angle in degrees
        fov_v: float = 60.0,  # vertical field of view half-angle in degrees
        dist_sigma: float = 0.01,  # standard deviation for distance noise (it scales proportionally to the object's distance)
        angle_sigma: float = 0.5,  # standard deviation for angle noise (azimuth and elevation) in degrees
        fn_rate: float = 0.01,  # chance to lose a perfectly visible object (False Negative)
        fp_rate: float = 0.001,  # chance PER FRAME to hallucinate a non-existent object
        confusion_rate: float = 0.05,  # chance to misclassify an object (e.g., Goal as Ball)
        send_unique_class_names: bool = True,  # If True, abbreviates world object names to a single uppercase letter. # noqa: FBT001, FBT002
        decimal_position_precision: int = 2,  # Number of decimal places to truncate coordinates to.
        max_number_of_false_positives: int = 5, # The absolute maximum number of hallucinations (fake objects) the camera can generate per frame.
    ):
        """
        Initializes the OfficialVisionGenerator with specific configuration parameters.
        These parameters control the camera constraints and the statistical error models.
        """
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.dist_sigma = dist_sigma
        self.angle_sigma = angle_sigma
        self.fn_rate = fn_rate
        self.fp_rate = fp_rate
        self.confusion_rate = confusion_rate
        self.send_unique_class_names = send_unique_class_names
        self.decimal_position_precision = decimal_position_precision
        self.max_number_of_false_positives = max_number_of_false_positives

    def filter_fov(self, v_sensor: Camera) -> np.ndarray:
        """
        Filters objects based on the camera's horizontal and vertical Field of View (FOV).
        Returns a boolean mask where True means the object is within the FOV boundaries.
        """
        return (v_sensor.azimuths >= -self.fov_h) & (v_sensor.azimuths <= self.fov_h) & (v_sensor.elevations >= -self.fov_v) & (v_sensor.elevations <= self.fov_v)

    def apply_false_negatives(self, visibility_mask: np.ndarray) -> np.ndarray:
        """
        Applies probability for the robot to miss an object.
        This simulates failures in object detection algorithms (like YOLO).
        """
        new_mask = visibility_mask.copy()
        n_items = len(new_mask)

        # Failure to detect VISIBLE objects
        if self.fn_rate > 0:
            drops = np.random.random(n_items) < self.fn_rate
            new_mask[new_mask & drops] = False

        return new_mask

    def generate_false_positives(self, all_world_names: np.ndarray) -> list[ObjectDetection]:
        """
        Simulates neural network hallucinations by generating fake objects (false positives)
        at random coordinates within the camera's Field of View.
        """
        obj_detections: list[ObjectDetection] = []

        # Determine how many false positives occur in this frame based on probability
        n_false_positives = np.sum(np.random.random(self.max_number_of_false_positives) < self.fp_rate)

        # If no hallucination triggered, return an empty list immediately
        if n_false_positives == 0:
            return obj_detections

        # Generate random polar coordinates within the FOV limits and a valid distance range (0.5m to 15m)
        fake_azis = np.random.uniform(-self.fov_h, self.fov_h, n_false_positives)
        fake_eles = np.random.uniform(-self.fov_v, self.fov_v, n_false_positives)
        fake_dists = np.random.uniform(0.5, 15.0, n_false_positives)

        # Randomly pick the classes/names for these fake objects
        fake_names = np.random.choice(all_world_names, size=n_false_positives)

        for name, azi, ele, dist in zip(fake_names, fake_azis, fake_eles, fake_dists, strict=False):
            final_name = name
            if self.send_unique_class_names:
                final_name = name[0].upper()
            obj_detections.append(ObjectDetection(final_name, azi, ele, dist))

        return obj_detections

    def apply_noise(self, distances: np.ndarray, azimuths: np.ndarray, elevations: np.ndarray) -> tuple:
        """
        Applies Gaussian noise to the polar coordinates (distance, azimuth, elevation)
        of the visible objects to simulate real-world sensor inaccuracies.
        """
        n_visible = len(distances)
        if n_visible == 0:
            return distances, azimuths, elevations

        # Generate Gaussian noise based on the configured standard deviations.
        # Distance noise is proportional to the object's actual distance (further = noisier).
        dist_noise = np.random.normal(0.0, distances * self.dist_sigma, n_visible)
        azi_noise = np.random.normal(0.0, self.angle_sigma, n_visible)
        ele_noise = np.random.normal(0.0, self.angle_sigma, n_visible)

        # Add noise to the original values. np.maximum ensures distance doesn't drop below 0.
        noisy_dists = np.maximum(0.0, distances + dist_noise)
        noisy_azis = azimuths + azi_noise
        noisy_eles = elevations + ele_noise

        return noisy_dists, noisy_azis, noisy_eles

    def _trunc(self, values: np.ndarray) -> np.ndarray:
        """
        Helper function to truncate decimal values to the specified precision.
        """
        factor = 10**self.decimal_position_precision
        return cast('np.ndarray', np.trunc(values * factor) / factor)

    def generate(self, v_sensor: Camera, agent: Any, agents_list: list) -> VisionPerception:
        """
        The main 'Template Method' that orchestrates the entire vision pipeline:
        FOV filtering, occlusion handling, false negatives, noise, and formatting.
        """
        if v_sensor is None:
            return VisionPerception('See', [])

        # 1. Evaluate Field of View (FOV)
        fov_mask = self.filter_fov(v_sensor)
        observer_id = agents_list.index(agent)

        # 2. Filter out occluded objects and the robot's own body parts
        not_occluded = ~v_sensor.is_occluded
        visibility_mask = fov_mask & not_occluded & (v_sensor.owner_ids != observer_id)

        # 3. Apply neural network failures (False Negatives)
        final_mask = self.apply_false_negatives(visibility_mask)

        obj_detections: list[PObjectDetection] = []
        n_visible = np.count_nonzero(final_mask)

        # Fetch the list of all possible world object names for confusion/hallucination logic
        all_world_names = np.unique(v_sensor.marker_names[v_sensor.owner_ids == -1])

        # ---------------------------------------------------------------------
        # 4. FALSE POSITIVES (Hallucinations)
        # ---------------------------------------------------------------------
        if self.fp_rate > 0 and len(all_world_names) > 0:
            false_positives = self.generate_false_positives(all_world_names)
            obj_detections.extend(false_positives)

        # Process REAL visible objects, if any survived the filters:
        if n_visible > 0:
            # Extract only the pure data that survived
            v_dists = v_sensor.distances[final_mask]
            v_azis = v_sensor.azimuths[final_mask]
            v_eles = v_sensor.elevations[final_mask]
            v_names = v_sensor.marker_names[final_mask]
            v_owners = v_sensor.owner_ids[final_mask]

            # 5. Apply Noise
            n_dists, n_azis, n_eles = self.apply_noise(v_dists, v_azis, v_eles)

            # Truncate floats to save network bandwidth
            n_dists = self._trunc(n_dists)
            n_azis = self._trunc(n_azis)
            n_eles = self._trunc(n_eles)

            # Create a mask to separate world static objects from dynamic agents
            world_mask = v_owners == -1

            # 6a. Real World Objects (With CLASS CONFUSION)
            if np.any(world_mask):
                for name, azi, ele, dist in zip(v_names[world_mask], n_azis[world_mask], n_eles[world_mask], n_dists[world_mask], strict=False):
                    final_name = name
                    if self.confusion_rate > 0 and len(all_world_names) > 1 and random.random() < self.confusion_rate:
                        wrong_names = [n for n in all_world_names if n != name]
                        final_name = random.choice(wrong_names)

                    if self.send_unique_class_names:
                        final_name = final_name[0].upper()

                    obj_detections.append(ObjectDetection(final_name, azi, ele, dist))

        # ---------------------------------------------------------------------
        # 6b. RANDOM SHIFT
        # ---------------------------------------------------------------------
        if len(obj_detections) > 0:
            n_shift = np.random.randint(0, len(obj_detections))
            obj_detections = obj_detections[-n_shift:] + obj_detections[:-n_shift]

        # ---------------------------------------------------------------------
        # 6c. PLAYER PARTS
        # ---------------------------------------------------------------------
        if n_visible > 0:
            unique_players = np.unique(v_owners[~world_mask])
            for p_idx in unique_players:
                p_mask = v_owners == p_idx
                target_agent = agents_list[p_idx]

                parts = []
                for name, azi, ele, dist in zip(v_names[p_mask], n_azis[p_mask], n_eles[p_mask], n_dists[p_mask], strict=False):
                    parts.append(ObjectDetection(name, azi, ele, dist))

                obj_detections.append(AgentDetection('P', target_agent.team_name, target_agent.agent_id.player_no, parts))

        return VisionPerception('See', obj_detections)
