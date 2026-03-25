import logging
from typing import Any

import mujoco
import numpy as np

from rcsssmj.sim.sensors import Camera
from rcsssmj.sim.vision_data import VisionData

logger = logging.getLogger(__name__)


def v_compile(mj_spec: Any) -> VisionData:
    """Extracts cameras, markers, and pre-allocates memory during compilation."""
    sensors = {site.name: Camera(site.name, site.name) for site in mj_spec.sites if site.name.endswith('camera')}

    agent_prefixes = [site.name[:-6] for site in mj_spec.sites if site.name.endswith('camera')]

    obj_markers = []
    owner_ids_list = []

    for site in mj_spec.sites:
        if site.name.endswith('-vismarker'):
            is_agent = False
            for prefix in agent_prefixes:
                if site.name.startswith(prefix):
                    marker_name = site.name[len(prefix):-10]
                    obj_markers.append((site.name, marker_name))
                    owner_ids_list.append(prefix)
                    is_agent = True
                    break

            if not is_agent:
                marker_name = site.name[:-10]
                obj_markers.append((site.name, marker_name))
                owner_ids_list.append('')

    n_world_markers = sum(1 for o in owner_ids_list if o == '')
    marker_sites = [m[0] for m in obj_markers]
    marker_names = np.array([m[1] for m in obj_markers], dtype=str)
    owner_ids = np.array(owner_ids_list, dtype=object)

    return VisionData(sensors, marker_sites, marker_names, owner_ids, n_world_markers)


def v_recompile(mj_spec: Any, _old_data: VisionData) -> VisionData:
    """Recompiles vision sensors and marker memory when the world changes."""
    return v_compile(mj_spec)


def v_step(v_data: VisionData, mj_model: Any, mj_data: Any, check_occlusion: bool) -> None:  # noqa: FBT001
    """Updates object positions in pre-allocated memory and calculates Ground Truth."""
    n_markers = len(v_data.marker_sites)

    # 1. Update object positions into pre-allocated memory directly
    for idx, site_name in enumerate(v_data.marker_sites):
        v_data.obj_pos[:, idx] = mj_data.site(site_name).xpos.astype(np.float64)

    # Collision & Visual Geom Groups (Enables checking against both physics and visual layers)
    geomgroup = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)

    # 2. Update the vision for each camera directly
    for camera_site_name, cam in v_data.sensors.items():
        observer_prefix = camera_site_name[:-6]
        logger.debug("--- Processing vision for observer: '%s' ---", observer_prefix)

        # exclude the observer's head to prevent immediate self-collision at the ray's origin.
        observer_head_name = observer_prefix + 'H2'
        bodyexclude_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, observer_head_name)

        s_site = mj_data.site(cam.site)
        camera_pos = s_site.xpos.astype(np.float64)
        camera_rot = s_site.xmat.astype(np.float64).reshape((3, 3))

        local_obj_pos = np.matmul(camera_rot.T, v_data.obj_pos - camera_pos[:, np.newaxis])
        distances = np.linalg.norm(local_obj_pos, axis=0)
        azimuths = np.degrees(np.atan2(local_obj_pos[1], local_obj_pos[0]))
        z_ratio = np.clip(local_obj_pos[2] / np.maximum(distances, 1e-6), -1.0, 1.0)
        elevations = np.degrees(np.asin(z_ratio))

        is_occluded_mask = np.full(n_markers, False, dtype=np.bool_)

        # --- NATIVE MUJOCO RAYCASTING ---
        if check_occlusion:
            geomid_arr = np.zeros(1, dtype=np.int32)

            # Test occlusion for each world object (the Target)
            for i in range(n_markers):
                target_owner_prefix = v_data.owner_ids[i]
                target_name = v_data.marker_names[i]

                ray_vec = v_data.obj_pos[:, i] - camera_pos
                ray_length = np.linalg.norm(ray_vec)

                # Skip if the object is exactly at the camera's origin
                if ray_length < 1e-4:
                    continue

                ray_dir = ray_vec / ray_length

                # Perform a single raycast from the camera to the target
                dist = mujoco.mj_ray(mj_model, mj_data, camera_pos, ray_dir, geomgroup, 0, bodyexclude_id, geomid_arr)

                # Check if we hit something before reaching the target (with a 5cm tolerance)
                if 0 <= dist < ray_length - 0.05:
                    hit_geom = geomid_arr[0]
                    if hit_geom != -1:
                        hit_body = mj_model.geom_bodyid[hit_geom]
                        hit_body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, hit_body)

                        hit_target = False

                        # fetch the body ID to which the target marker (site) is physically attached
                        target_site_name = v_data.marker_sites[i]
                        target_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, target_site_name)
                        target_body_id = mj_model.site_bodyid[target_site_id]

                        if hit_body == target_body_id:
                            hit_target = True

                        if not hit_target:
                            logger.debug(
                                "[%s] TARGET OCCLUDED: '%s' (owner: '%s') was blocked by '%s'",
                                observer_prefix, target_name, target_owner_prefix, hit_body_name
                            )
                            is_occluded_mask[i] = True
                            
                        else:
                            logger.debug(
                                "[%s] TARGET VISIBLE: '%s' (owner: '%s') - hit target's own body '%s'",
                                observer_prefix, target_name, target_owner_prefix, hit_body_name
                            )

                else:
                    logger.debug(
                        "[%s] TARGET VISIBLE: '%s' (owner: '%s') - clear line of sight",
                        observer_prefix, target_name, target_owner_prefix
                    )

        # --- SENSOR UPDATE ---
        cam.marker_names = v_data.marker_names
        cam.distances = distances
        cam.azimuths = azimuths
        cam.elevations = elevations
        cam.is_occluded = is_occluded_mask
        cam.owner_ids = v_data.owner_ids