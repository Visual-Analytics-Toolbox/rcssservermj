import logging
import tempfile
import threading
import queue
import os
from collections.abc import Sequence

import mujoco
import msgpack

from rcsssmj.games.soccer.sim.soccer_state_info import SoccerGameInformation
from rcsssmj.server.remote_monitor import SimMonitor
from rcsssmj.sim.state_info import SceneGraph, SimStateInformation

logger = logging.getLogger(__name__)

class MsgpackLoggerMonitor(SimMonitor):
    """Direct file logger monitor for simulation using msgpack."""

    def __init__(self, filename: str = "simulation_log.dat") -> None:
        super().__init__(update_interval=1)

        self.filename = filename
        self.file = open(self.filename, 'wb')
        self.model = None
        self.game_state: SoccerGameInformation | None = None
        
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()
        
        logger.info(f"MsgpackLoggerMonitor initializing... Logging simulation data to {self.filename}")

    def _writer_thread(self) -> None:
        """Background thread handling all file I/O to avoid blocking the simulation loop via the GIL."""
        tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
        
        while True:
            item = self._queue.get()
            if item is None:
                break
                
            msg_type = item.get("type")
            
            try:
                if msg_type == "model_request":
                    model = item["model"]
                    try:
                        with tempfile.NamedTemporaryFile(dir=tmp_dir, mode='w+', delete=True, suffix='.xml') as tmp:
                            mujoco.mj_saveLastXML(tmp.name, model)
                            tmp.seek(0)
                            model_xml = tmp.read()
                        
                        msg = msgpack.packb({"type": "model", "xml": model_xml})
                    except Exception:
                        with tempfile.NamedTemporaryFile(dir=tmp_dir, mode='w+b', delete=True, suffix='.mjb') as tmp:
                            mujoco.mj_saveModel(model, tmp.name)
                            tmp.seek(0)
                            model_mjb = tmp.read()
                        msg = msgpack.packb({"type": "model_mjb", "mjb": model_mjb})
                        
                    self.file.write(len(msg).to_bytes(4, byteorder='little'))
                    self.file.write(msg)
                    self.file.flush()
                    
                elif msg_type == "state_data":
                    state_dict = {
                        "type": "state",
                        "frame": item["frame"],
                        "time": item["time"],
                        "qpos": item["qpos"].tolist(),
                        "qvel": item["qvel"].tolist()
                    }
                    if item["game_state"]:
                        gs = item["game_state"]
                        state_dict["game_state"] = {
                            "left_team": gs.left_team,
                            "right_team": gs.right_team,
                            "left_score": gs.left_score,
                            "right_score": gs.right_score,
                            "play_time": gs.play_time,
                            "play_mode": gs.play_mode,
                        }
                    
                    msg = msgpack.packb(state_dict)
                    self.file.write(len(msg).to_bytes(4, byteorder='little'))
                    self.file.write(msg)
            except Exception as e:
                logger.error(f"Writer thread error: {e}")

    def shutdown(self) -> None:
        super().shutdown()
        
        # Stop the writer thread
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=2.0)
            
        if self.file and not self.file.closed:
            self.file.flush()
            self.file.close()

    def update(self, state_info: Sequence[SimStateInformation], frame_id: int) -> None:
        scene_graph = None
        for info in state_info:
            if isinstance(info, SoccerGameInformation):
                self.game_state = info
            elif isinstance(info, SceneGraph):
                scene_graph = info

        if scene_graph is not None:
            #Check for Model Change
            if self.model is not scene_graph.mj_model:
                self.model = scene_graph.mj_model
                # Request model snapshot in background
                self._queue.put({"type": "model_request", "model": self.model})

            # Extract state data
            # Copy numpy arrays immediately to avoid thread-safety issues if main thread modifies them
            data = scene_graph.mj_data
            item = {
                "type": "state_data",
                "frame": frame_id,
                "time": data.time,
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "game_state": self.game_state
            }
            self._queue.put(item)

