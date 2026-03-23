import logging
import tempfile
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
        logger.info(f"MsgpackLoggerMonitor initializing... Logging simulation data to {self.filename}")

    def shutdown(self) -> None:
        super().shutdown()
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
            if self.model is not scene_graph.mj_model:
                self.model = scene_graph.mj_model

                # Get the authoritative XML copy of the model
                try:
                    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.xml') as tmp:
                        mujoco.mj_saveLastXML(tmp.name, self.model)
                        tmp.seek(0)
                        model_xml = tmp.read()
                    
                    msg = msgpack.packb({
                        "type": "model",
                        "xml": model_xml
                    })
                except mujoco.FatalError:
                    with tempfile.NamedTemporaryFile(mode='w+b', delete=True, suffix='.mjb') as tmp:
                        mujoco.mj_saveModel(self.model, tmp.name)
                        tmp.seek(0)
                        model_mjb = tmp.read()
                    
                    msg = msgpack.packb({
                        "type": "model_mjb",
                        "mjb": model_mjb
                    })
                
                # write length-prefix then binary msgpack string
                self.file.write(len(msg).to_bytes(4, byteorder='little'))
                self.file.write(msg)
                self.file.flush()

            data = scene_graph.mj_data
            
            state_dict = {
                "type": "state",
                "frame": frame_id,
                "time": data.time,
                "qpos": data.qpos.tolist(),
                "qvel": data.qvel.tolist()
            }
            
            if self.game_state is not None:
                state_dict["game_state"] = {
                    "left_team": self.game_state.left_team,
                    "right_team": self.game_state.right_team,
                    "left_score": self.game_state.left_score,
                    "right_score": self.game_state.right_score,
                    "play_time": self.game_state.play_time,
                    "play_mode": self.game_state.play_mode,
                }
                
            msg = msgpack.packb(state_dict)
            self.file.write(len(msg).to_bytes(4, byteorder='little'))
            self.file.write(msg)
