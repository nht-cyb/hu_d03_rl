"""Convert an HU_D03 motion CSV to the .npz that mjlab's tracking task reads.

Same as mjlab's scripts/csv_to_npz.py (which hard-codes the G1), but for the
HU_D03, and it writes the .npz to a local path instead of requiring a Weights &
Biases registry upload. Train with:

  train Mjlab-Tracking-Flat-LimX-HU-D03 --env.commands.motion.motion-file motion.npz

  python scripts/csv_to_npz_hu_d03.py --input-file walk_hu_d03.csv \
      --output-file walk_hu_d03.npz [--input-fps 30] [--output-fps 50]
"""

from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab  # noqa: F401  (configures Warp, loads the HU_D03 tasks)
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg

from hu_d03_mjlab.tasks.tracking_env_cfgs import hu_d03_flat_tracking_env_cfg

LOG_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def main(
  input_file: str,
  output_file: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  line_range: tuple[int, int] | None = None,
):
  """Args:
    line_range: Only convert these CSV lines (1-based, inclusive), e.g. (1, 1500).
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(hu_d03_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  # CSV columns are the HU_D03 joints in model order.
  joint_names = list(robot.joint_names)

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=int(input_fps),
    output_fps=int(output_fps),
    device=sim.device,
    line_range=line_range,
  )
  assert motion.motion_dof_poss_input.shape[1] == len(joint_names), (
    f"CSV has {motion.motion_dof_poss_input.shape[1]} joints, robot has "
    f"{len(joint_names)}"
  )
  joint_ids = robot.find_joints(joint_names, preserve_order=True)[0]

  log: dict[str, list] = {k: [] for k in LOG_KEYS}
  scene.reset()

  for _ in tqdm(range(motion.output_frames), desc="frames"):
    (
      (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel),
      _,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_ids] = dof_pos
    joint_vel[:, joint_ids] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    data = robot.data
    log["joint_pos"].append(data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(data.body_link_ang_vel_w[0].cpu().numpy().copy())

  out = {"fps": [output_fps], **{k: np.stack(v, axis=0) for k, v in log.items()}}
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_file, **out)
  print(f"wrote {output_file} ({len(log['joint_pos'])} frames at {output_fps} fps)")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
