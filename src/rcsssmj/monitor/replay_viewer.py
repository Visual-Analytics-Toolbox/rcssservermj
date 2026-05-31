import argparse
import time
import mujoco
import msgpack

from rcsssmj.games.soccer.sim.soccer_state_info import SoccerGameInformation
from rcsssmj.monitor.mujoco_monitor import MujocoMonitor

def main() -> None:
    parser = argparse.ArgumentParser(description="Replay viewer for MuJoCo simulation.")
    parser.add_argument("logfile", type=str, help="Input file (e.g. simulation_log.dat)")
    parser.add_argument("--fps", type=int, default=100, help="Frames per second for replay")
    args = parser.parse_args()

    model = None
    data = None
    monitor = None

    print(f"Reading from {args.logfile}...")
    
    delay = 1.0 / args.fps

    with open(args.logfile, "rb") as f:
        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                break
            msg_len = int.from_bytes(length_bytes, byteorder='little')
            msg_bytes = f.read(msg_len)
            
            if len(msg_bytes) != msg_len:
                break
                
            msg = msgpack.unpackb(msg_bytes)
            
            if msg["type"] == "model":
                xml_str = msg["xml"].decode('utf-8') if isinstance(msg["xml"], bytes) else msg["xml"]
                model = mujoco.MjModel.from_xml_string(xml_str)
                data = mujoco.MjData(model)
                if monitor is None:
                    monitor = MujocoMonitor(model, render_interval=1)
                else:
                    monitor.model = model
                    monitor.scene = mujoco.MjvScene(model, 1000)
                    monitor.context = mujoco.MjrContext(model, mujoco.mjtFontScale(mujoco.mjtFontScale.mjFONTSCALE_150))
            elif msg["type"] == "model_mjb":
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w+b', delete=True, suffix='.mjb') as tmp:
                    tmp.write(msg["mjb"])
                    tmp.flush()
                    model = mujoco.MjModel.from_binary_path(tmp.name)
                data = mujoco.MjData(model)
                if monitor is None:
                    monitor = MujocoMonitor(model, render_interval=1)
                else:
                    monitor.model = model
                    monitor.scene = mujoco.MjvScene(model, 1000)
                    monitor.context = mujoco.MjrContext(model, mujoco.mjtFontScale(mujoco.mjtFontScale.mjFONTSCALE_150))
            elif msg["type"] == "state":
                if model is None or monitor is None or data is None:
                    continue
                    
                data.time = msg["time"]
                data.qpos[:] = msg["qpos"]
                data.qvel[:] = msg["qvel"]
                
                mujoco.mj_forward(model, data)
                
                game_state_dict = msg.get("game_state")
                if game_state_dict:
                    monitor.game_state = SoccerGameInformation(
                        left_team=game_state_dict["left_team"],
                        right_team=game_state_dict["right_team"],
                        left_score=game_state_dict["left_score"],
                        right_score=game_state_dict["right_score"],
                        play_time=game_state_dict["play_time"],
                        play_mode=game_state_dict["play_mode"]
                    )
                
                monitor.render(data)
                
                time.sleep(delay)

    print("Replay finished. Holding last frame...")
    if monitor is not None and data is not None:
        try:
            while True:
                monitor.render(data)
                time.sleep(1/30)
        except KeyboardInterrupt:
            print("Closing...")
            
    if monitor is not None:
        monitor.shutdown()

if __name__ == "__main__":
    main()
