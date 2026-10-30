"""Roll out a policy headless and save an .mp4 (for Colab/Kaggle, where mjlab's
`play` viewer cannot open a window).

  python scripts/record_video.py Mjlab-Velocity-Flat-LimX-HU-D03 \
      --checkpoint-file logs/.../model_5000.pt --output-file walk.mp4
  python scripts/record_video.py Mjlab-Tracking-Flat-LimX-HU-D03 \
      --checkpoint-file ... --motion-file motion.npz --output-file mimic.mp4
  # --agent zero holds the default pose (stand test, no checkpoint needed).
  # --start-seconds 42 starts a tracking video at that time of the motion (e.g. one dance of a medley).
  # Defaults to 1280x720; e.g. --width 1920 --height 1080 --distance 4 for 1080p.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import mediapy
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg


def main(
  task_id: tyro.conf.Positional[str],
  output_file: str,
  checkpoint_file: str | None = None,
  motion_file: str | None = None,
  agent: Literal["trained", "zero"] = "trained",
  seconds: float = 10.0,
  start_seconds: float = 0.0,
  no_terminations: bool = False,
  width: int = 1280,
  height: int = 720,
  distance: float = 3.0,
  elevation: float = -15.0,
  azimuth: float = 90.0,
):
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1
  # mjlab renders 320x240 from 5 m away by default; frame the whole robot instead.
  env_cfg.viewer.width, env_cfg.viewer.height = width, height
  env_cfg.viewer.distance = distance
  env_cfg.viewer.elevation = elevation
  env_cfg.viewer.azimuth = azimuth
  if no_terminations:
    env_cfg.terminations = {}

  motion_cmd = env_cfg.commands.get("motion")
  if isinstance(motion_cmd, MotionCommandCfg):
    if motion_file is None:
      raise ValueError("Tracking tasks need --motion-file.")
    motion_cmd.motion_file = motion_file

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if agent == "zero":
    shape = env.unwrapped.action_space.shape

    def policy(obs):
      del obs
      return torch.zeros(shape, device=env.unwrapped.device)
  else:
    if checkpoint_file is None:
      raise ValueError("--checkpoint-file is required with --agent trained.")
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint_file, load_cfg={"actor": True}, strict=True,
                map_location=device)
    policy = runner.get_inference_policy(device=device)

  if start_seconds > 0:
    # Put the robot exactly in the reference state at that frame (play mode starts at frame 0).
    command = env.unwrapped.command_manager.get_term("motion")
    frame = round(start_seconds / env.unwrapped.step_dt)
    command.reset_to_frame(torch.arange(env.num_envs, device=env.unwrapped.device), frame)
    env.unwrapped.sim.forward()
    command.update_relative_body_poses()

  obs = env.get_observations()
  frames = []
  steps = int(seconds / env.unwrapped.step_dt)
  with torch.inference_mode():
    for _ in range(steps):
      obs, _, _, _ = env.step(policy(obs))
      frames.append(env.unwrapped.render())

  fps = round(1.0 / env.unwrapped.step_dt)
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(output_file, frames, fps=fps)
  print(f"wrote {output_file} ({len(frames)} frames at {fps} fps)")
  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
