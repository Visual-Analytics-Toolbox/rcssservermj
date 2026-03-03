from enum import IntEnum
from typing import Any
import numpy as np

from rcsssmj.sim.sensors import Camera
from rcsssmj.sim.vision_data import VisionData

# file handler
import logging
fh = logging.FileHandler(filename='debug.log', mode='w')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
fh.setLevel(logging.DEBUG)

# configure logging
logging.basicConfig(handlers=[fh], level=logging.DEBUG)
# ---------- LOGGING CONFIG ----------

logger = logging.getLogger(__name__)

class OcclusionState(IntEnum):
    NONE = 0
    PARTIAL = 1
    FULL = 2

def v_compile(mj_spec: Any) -> VisionData:
    """Extracts all cameras from the spec."""
    sensors = {site.name: Camera(site.name, site.name) for site in mj_spec.sites if site.name.endswith('camera')}
    return VisionData(sensors)

def v_recompile(mj_spec: Any, old_data: VisionData) -> VisionData:
    """
    Recompiles vision sensors when the world changes (e.g., agents entering the game).
    Since vision is recalculated from scratch every frame (no time persistence), 
    we just extract the sensors again.
    """
    return v_compile(mj_spec)

def v_forward(v_data: VisionData, mj_data: Any, agents: list, obj_pos: np.ndarray, obj_radii: np.ndarray, obj_markers: list, n_world_markers: int, check_occlusion: bool) -> None:
    """Calculates the Ground Truth vision data for each camera."""
    n_markers = len(obj_markers)
    marker_names = [m[0] for m in obj_markers]

    logger.debug(marker_names)
    
    # 1. Prepare the "owners" mask ONLY ONCE
    # Creates an array where each index tells us who the marker belongs to.
    # -1 means a world object (ball, goal), 0..N means the Agent's ID.
    owner_ids = np.full(n_markers, -1, dtype=int)
    cursor = n_world_markers
    for idx_a, p in enumerate(agents):
        n_p_markers = len(p.markers)
        owner_ids[cursor : cursor + n_p_markers] = idx_a
        cursor += n_p_markers

    # 2. Update the vision for each agent (Camera)
    for agent_idx, agent in enumerate(agents):
        camera_site_name = agent.agent_id.prefix + 'camera'
        cam = v_data.sensors.get(camera_site_name, None)
        if cam is None:
            continue
        
        # Extract the camera's position (x,y,z) and rotation matrix in the world
        s_site = mj_data.site(cam.site)
        camera_pos = s_site.xpos.astype(np.float64)
        camera_rot = s_site.xmat.astype(np.float64).reshape((3, 3))

        # --- COORDINATE TRANSFORMATION ---
        # Move the world origin to the camera and rotate to the camera's local frame
        local_obj_pos = np.matmul(camera_rot.T, obj_pos - camera_pos[:, np.newaxis])
        
        # Convert Cartesian (X, Y, Z) to Spherical/Polar (Distance, Azimuth, Elevation)
        distances = np.linalg.norm(local_obj_pos, axis=0)
        azimuths = np.degrees(np.atan2(local_obj_pos[1], local_obj_pos[0]))
        z_ratio = np.clip(local_obj_pos[2] / np.maximum(distances, 1e-6), -1.0, 1.0)
        elevations = np.degrees(np.asin(z_ratio))

        # Initialize all objects as visible (not occluded)
        occlusion_states = np.full(n_markers, OcclusionState.NONE.value, dtype=np.int8)
        
        # --- VECTORIZED OCCLUSION LOGIC ---
        if check_occlusion:
            observer_id = agent_idx # ID of the robot that is currently looking
            
            # Test occlusion for each world object (the Target)
            for i in range(n_markers):
                target_owner = owner_ids[i]
                target_name = marker_names[i]

                # Vector from the camera to the Target (Line of Sight)
                ray_vec = obj_pos[:, i] - camera_pos
                ray_length = np.linalg.norm(ray_vec)

                # Prevent math errors if the object is exactly at the camera's center
                if ray_length < 1e-4:
                    continue
                    
                # Unit vector (gaze direction)
                ray_dir = ray_vec / ray_length
                
                # Vectors from the camera to ALL other world objects (Potential Obstacles)
                obst_vecs = obj_pos - camera_pos[:, np.newaxis]
                
                # Project obstacles onto the Line of Sight using Dot Product (optimized with einsum)
                projs = np.einsum('i,ij->j', ray_dir, obst_vecs) 

                # Filter only the obstacles that actually matter
                valid_obst_mask = (projs > 0.05) & (projs < ray_length - 0.05) & \
                                (owner_ids != observer_id) & \
                                (owner_ids != target_owner) & \
                                (owner_ids != -1)
                
                if not np.any(valid_obst_mask):
                    continue

                valid_obst_vecs = obst_vecs[:, valid_obst_mask]
                valid_obst_radii = obj_radii[valid_obst_mask]
                valid_projs = projs[valid_obst_mask]
                
                # Calculate the "Perpendicular Vector" from the obstacle to the Line of Sight
                perp_vecs = valid_obst_vecs - np.outer(ray_dir, valid_projs)
                perp_dists = np.linalg.norm(perp_vecs, axis=0)

                # It considers an obstacle if it is within a distance smaller than its radius + 20% margin.
                close_mask = perp_dists < (valid_obst_radii * 1.2)

                if not np.any(close_mask):
                    continue

                close_perp_vecs = perp_vecs[:, close_mask]
                close_dists = perp_dists[close_mask]
                close_radii = valid_obst_radii[close_mask]
                n_close = len(close_dists)

                # --- OCCLUSION RESOLUTION ---
                
                # 1. Hard Occlusion: Blocks the majority of the object
                if np.any(close_dists < (close_radii * 0.8)):
                    occlusion_states[i] = OcclusionState.FULL.value
                    logger.debug(f"[{camera_site_name}] Hard Occlusion: '{target_name}' is FULLY occluded by an obstacle.")
                    continue
                    
                # 2. Flanking (Multiple obstacles grazing the vision)
                elif n_close > 1:
                    dot_products = np.einsum('i,ij->j', close_perp_vecs[:, 0], close_perp_vecs[:, 1:])
                    
                    if np.any(dot_products < 0):
                        occlusion_states[i] = OcclusionState.FULL.value
                        logger.debug(f"[{camera_site_name}] Flanking Occlusion: '{target_name}' is FULLY occluded (squeezed between {n_close} obstacles).")
                    else:
                        occlusion_states[i] = OcclusionState.PARTIAL.value
                        logger.debug(f"[{camera_site_name}] Soft Occlusion: '{target_name}' is PARTIALLY occluded by multiple obstacles on the same side.")
                            
                # 3. Soft Occlusion: Grazing the outer edge
                elif n_close == 1:
                    occlusion_states[i] = OcclusionState.PARTIAL.value
                    logger.debug(f"[{camera_site_name}] Soft Occlusion: '{target_name}' is PARTIALLY occluded by a single obstacle.")
        
        # --- SENSOR UPDATE ---
        cam.marker_names = marker_names
        cam.distances = distances
        cam.azimuths = azimuths
        cam.elevations = elevations
        cam.occlusion_states = occlusion_states
        cam.owner_ids = owner_ids