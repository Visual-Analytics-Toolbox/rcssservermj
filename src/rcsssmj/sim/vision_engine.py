from typing import Any

import mujoco
import numpy as np

from rcsssmj.sim.sensors import Camera
from rcsssmj.sim.vision_data import VisionData


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

    geomgroup = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)

    # 2. Update the vision for each camera directly
    for camera_site_name, cam in v_data.sensors.items():
        observer_prefix = camera_site_name[:-6]

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

                if ray_length < 1e-4:
                    continue

                ray_dir = ray_vec / ray_length

                current_pnt = camera_pos.copy()
                remaining_dist = ray_length
                target_occluded = False

                # Piercing Raycast Loop: Ignore our own body parts
                while remaining_dist > 0.05: # 5cm tolerance near the target center
                    dist = mujoco.mj_ray(mj_model, mj_data, current_pnt, ray_dir, geomgroup, 0, -1, geomid_arr)
                    hit_geom = geomid_arr[0]

                    if dist < 0 or dist >= remaining_dist - 0.05:
                        # Clear path to target (or hit was behind the target)
                        break

                    if hit_geom != -1:
                        hit_body = mj_model.geom_bodyid[hit_geom]
                        hit_body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, hit_body)

                        # 1. If we hit ourselves, pierce through and continue
                        if hit_body_name and hit_body_name.startswith(observer_prefix):
                            advance = dist + 0.001 # Advance 1mm past the collision
                            current_pnt += ray_dir * advance
                            remaining_dist -= advance
                        else:
                            # We hit something! Let's verify if it's the Target's own body
                            hit_target = False

                            # Target is a Player
                            if target_owner_prefix != '':
                                if hit_body_name and hit_body_name.startswith(target_owner_prefix):
                                    hit_target = True
                            # Target is a World Object (Ball, Goal)
                            elif hit_body_name and target_name in hit_body_name:
                                hit_target = True

                            if not hit_target:
                                target_occluded = True
                            break
                    else:
                        break

                # Since mj_ray is binary (hit or not hit), we only use FULL occlusion now.
                if target_occluded:
                    is_occluded_mask[i] = True

        # --- SENSOR UPDATE ---
        cam.marker_names = v_data.marker_names
        cam.distances = distances
        cam.azimuths = azimuths
        cam.elevations = elevations
        cam.is_occluded = is_occluded_mask
        cam.owner_ids = v_data.owner_ids
