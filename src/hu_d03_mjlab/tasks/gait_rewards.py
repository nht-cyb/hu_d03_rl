"""Reward terms that shape a human-like gait without a reference motion or phase clock."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command_speed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])


def contact_pattern(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  move_threshold: float = 0.1,
  run_threshold: float = 1.8,
) -> torch.Tensor:
  """Reward the foot contacts of the commanded gait.

  Standing: both feet down. Walking: one foot down, so the legs alternate; double
  support earns nothing but is not punished, and it stays brief. Running: at most
  one foot down, which also allows the flight phase.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  num_down = (found > 0).sum(dim=1)
  speed = _command_speed(env, command_name)
  return torch.where(
    speed < move_threshold,
    (num_down == 2).float(),
    torch.where(speed < run_threshold, num_down == 1, num_down <= 1).float(),
  )


def gait_symmetry(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.3,
) -> torch.Tensor:
  """Penalize a limp: both legs should have the same swing and stance times."""
  sensor: ContactSensor = env.scene[sensor_name]
  air, contact = sensor.data.last_air_time, sensor.data.last_contact_time
  assert air is not None and contact is not None
  cost = torch.square(air[:, 0] - air[:, 1]) + torch.square(contact[:, 0] - contact[:, 1])
  return cost * (_command_speed(env, command_name) > command_threshold).float()


def arm_leg_swing(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  gain: float = 0.7,
  std: float = 0.25,
  command_threshold: float = 0.3,
) -> torch.Tensor:
  """Reward swinging each arm with the opposite leg.

  asset_cfg.joint_names must be (left shoulder pitch, right shoulder pitch, left hip
  pitch, right hip pitch), with preserve_order=True. On the HU_D03 a negative angle
  swings both the arm and the thigh forward, so the signs match. Angles are measured
  from the default pose.
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q = q - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  error = torch.square(q[:, 0] - gain * q[:, 3]) + torch.square(q[:, 1] - gain * q[:, 2])
  moving = _command_speed(env, command_name) > command_threshold
  return torch.exp(-0.5 * error / std**2) * moving.float()


def stance_width(
  env: ManagerBasedRlEnv,
  target: float,
  asset_cfg: SceneEntityCfg,
  std: float = 0.05,
) -> torch.Tensor:
  """Reward feet at a natural lateral distance, measured in the pelvis frame."""
  asset: Entity = env.scene[asset_cfg.name]
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids]
  between_w = feet[:, 0] - feet[:, 1]
  width = quat_apply_inverse(asset.data.root_link_quat_w, between_w)[:, 1]
  return torch.exp(-torch.square(width - target) / std**2)


def leg_extension(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  std: float = 0.04,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward walking tall instead of crouching.

  Height is measured from the lower foot, not the world, so it means the same thing
  on a slope or on stairs.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].min(dim=1).values
  height = asset.data.root_link_pos_w[:, 2] - foot_z
  moving = _command_speed(env, command_name) > command_threshold
  return torch.exp(-torch.square(height - target_height) / std**2) * moving.float()


def mechanical_power(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize mechanical power: low-energy motion tends to look more natural."""
  asset: Entity = env.scene["robot"]
  return torch.sum(torch.abs(asset.data.qfrc_actuator * asset.data.joint_vel), dim=1)
