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
                    marker_name = site.name[len(prefix) : -10]
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

    # Update object positions into pre-allocated memory directly
    for idx, site_name in enumerate(v_data.marker_sites):
        v_data.obj_pos[:, idx] = mj_data.site(site_name).xpos.astype(np.float64)

    # Collision & Visual Geom Groups (Enables checking against both physics and visual layers)
    geomgroup = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)

    # Pre-fetch target body ids for all markers once per step (fast list comprehension -> numpy)
    target_site_ids = np.array([mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name) for name in v_data.marker_sites], dtype=np.int32)
    target_body_ids = mj_model.site_bodyid[target_site_ids]

    # Update the vision for each camera directly
    for camera_site_name, cam in v_data.sensors.items():
        observer_prefix = camera_site_name[:-6]

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
            # Vectorized calculations for all rays
            ray_vecs = v_data.obj_pos.T - camera_pos  # (n_markers, 3)
            ray_lengths = np.linalg.norm(ray_vecs, axis=1)

            valid_mask = ray_lengths >= 1e-4

            ray_dirs = np.zeros_like(ray_vecs)
            ray_dirs[valid_mask] = ray_vecs[valid_mask] / ray_lengths[valid_mask, np.newaxis]

            # Flatten to ensure contiguous 1D array as expected by pybind
            ray_dirs_flat = np.ascontiguousarray(ray_dirs, dtype=np.float64).flatten()
            geomid_arr = np.full(n_markers, -1, dtype=np.int32)
            dist_arr = np.full(n_markers, -1.0, dtype=np.float64)

            # Cast all rays at once
            mujoco.mj_multiRay(mj_model, mj_data, camera_pos, ray_dirs_flat, geomgroup, 0, bodyexclude_id, geomid_arr, dist_arr, None, n_markers, -1.0)

            # Process raycast results completely vectorized
            hit_something_mask = (dist_arr >= 0) & (dist_arr < ray_lengths - 0.05)

            # Extract bodies only for valid geoms to avoid out-of-bounds indexing
            safe_geomid_arr = np.maximum(geomid_arr, 0)
            hit_bodies = mj_model.geom_bodyid[safe_geomid_arr]

            valid_geom_mask = geomid_arr != -1
            not_target_mask = hit_bodies != target_body_ids

            # Mask is True if ray is valid, hit a geom, and that geom is NOT the target body
            is_occluded_mask = valid_mask & hit_something_mask & valid_geom_mask & not_target_mask

        # --- SENSOR UPDATE ---
        cam.marker_names = v_data.marker_names
        cam.distances = distances
        cam.azimuths = azimuths
        cam.elevations = elevations
        cam.is_occluded = is_occluded_mask
        cam.owner_ids = v_data.owner_ids
