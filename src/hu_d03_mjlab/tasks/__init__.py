"""Registers the HU_D03 tasks with mjlab (loaded via the `mjlab.tasks` entry point)."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .rl_cfgs import hu_d03_tracking_ppo_runner_cfg, hu_d03_velocity_ppo_runner_cfg
from .tracking_env_cfgs import hu_d03_flat_tracking_env_cfg
from .velocity_env_cfgs import hu_d03_flat_env_cfg, hu_d03_rough_env_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-LimX-HU-D03",
  env_cfg=hu_d03_flat_env_cfg(),
  play_env_cfg=hu_d03_flat_env_cfg(play=True),
  rl_cfg=hu_d03_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-LimX-HU-D03",
  env_cfg=hu_d03_rough_env_cfg(),
  play_env_cfg=hu_d03_rough_env_cfg(play=True),
  rl_cfg=hu_d03_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-LimX-HU-D03",
  env_cfg=hu_d03_flat_tracking_env_cfg(),
  play_env_cfg=hu_d03_flat_tracking_env_cfg(play=True),
  rl_cfg=hu_d03_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-LimX-HU-D03-No-State-Estimation",
  env_cfg=hu_d03_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=hu_d03_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=hu_d03_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
