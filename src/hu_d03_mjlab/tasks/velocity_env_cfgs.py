"""LimX HU_D03 velocity environment configurations (adapted from mjlab's G1 ones)."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import config as terrain_cfg

from hu_d03_mjlab.robot.hu_d03_constants import (
  FOOT_GEOMS,
  HU_D03_ACTION_SCALE,
  get_hu_d03_robot_cfg,
)
from hu_d03_mjlab.tasks import gait_rewards

PELVIS = "base_link"
TORSO = "waist_pitch_link"


def _hu_d03_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Settings shared by the flat and rough tasks."""
  cfg = make_velocity_env_cfg()

  # mjlab's G1 uses 500 iterations and 70 contacts per env, too much for a T4's 15 GB. The
  # collision buffers need about 40 kB (at 300 iterations) times nconmax * num_envs, once
  # for each of the step and forward CUDA graphs: about 10 GB here, on top of about 2 GB.
  cfg.sim.mujoco.ccd_iterations = 300
  cfg.sim.contact_sensor_maxmatch = 500
  # A pool shared by all envs, not a per-env cap; envs use about 4 on average.
  cfg.sim.nconmax = 30

  cfg.scene.entities = {"robot": get_hu_d03_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = PELVIS

  site_names = ("left_foot", "right_foot")

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=PELVIS, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=PELVIS, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = HU_D03_ACTION_SCALE

  cfg.viewer.body_name = TORSO

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.75

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO,)

  # Same std rationale as the G1: loose knees/hip pitch, tight ankle roll and
  # waist roll/pitch, moderate arms. The HU_D03 also has a head and hand yaw.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist and head.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    r"head_.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
    r".*hand_yaw.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist and head.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    r"head_.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
    r".*hand_yaw.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = (TORSO,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TORSO,)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def hu_d03_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create LimX HU_D03 rough terrain velocity configuration with a human-like gait.

  Meant to be warm-started from a flat policy (scripts/widen_flat_checkpoint.py), so
  it starts on easy terrain and at speeds that policy already handles.
  """
  cfg = _hu_d03_base_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  # Terrains a person walks on; mjlab's 45 degree slopes would force an unnatural gait.
  terrain.sub_terrains = {
    "flat": terrain_cfg.flat(proportion=0.15),
    # Robots spawn at the center of a tile: a pyramid's top, or the bottom of an inverted one.
    "stairs_down": terrain_cfg.pyramid_stairs(
      proportion=0.2, step_height_range=(0.02, 0.15), step_width=0.35
    ),
    "stairs_up": terrain_cfg.pyramid_stairs_inv(
      proportion=0.2, step_height_range=(0.02, 0.15), step_width=0.35
    ),
    "slope_down": terrain_cfg.hf_pyramid_slope(proportion=0.1, slope_range=(0.0, 0.45)),
    "slope_up": terrain_cfg.hf_pyramid_slope_inv(
      proportion=0.1, slope_range=(0.0, 0.45)
    ),
    "random_rough": terrain_cfg.random_rough(proportion=0.1, noise_range=(0.01, 0.06)),
    "waves": terrain_cfg.wave_terrain(proportion=0.1, amplitude_range=(0.0, 0.12)),
    "obstacles": terrain_cfg.discrete_obstacles(
      proportion=0.05, obstacle_height_range=(0.02, 0.12), num_obstacles=20
    ),
  }
  cfg.scene.terrain.max_init_terrain_level = 2

  # People walk forward far more than they strafe or back up, and they also run.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_forward_envs = 0.4
  twist_cmd.ranges.lin_vel_x = (-0.8, 1.5)
  twist_cmd.ranges.lin_vel_y = (-0.4, 0.4)
  if not play:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.8, 1.5), "ang_vel_z": (-0.5, 0.5)},
      {"step": 1500 * 24, "lin_vel_x": (-1.0, 2.0), "ang_vel_z": (-0.7, 0.7)},
      {"step": 4000 * 24, "lin_vel_x": (-1.0, 3.0), "lin_vel_y": (-0.5, 0.5)},
    ]

  # Tighter on the joints that made the flat gait stiff-armed and wide-hipped, looser
  # on the shoulder pitch and waist yaw so that the arms and torso can counter-swing.
  pose = cfg.rewards["pose"].params
  pose["std_walking"] |= {
    r".*hip_roll.*": 0.12,
    r".*hip_yaw.*": 0.1,
    r".*shoulder_pitch.*": 0.4,
    r".*shoulder_roll.*": 0.08,
    r".*shoulder_yaw.*": 0.08,
    r".*elbow.*": 0.2,
    r"head_.*": 0.05,
    r".*waist_yaw.*": 0.25,
  }
  pose["std_running"] |= {
    r".*hip_yaw.*": 0.15,
    r".*shoulder_pitch.*": 0.7,
    r".*shoulder_roll.*": 0.12,
    r".*shoulder_yaw.*": 0.12,
    r".*elbow.*": 0.5,
    r"head_.*": 0.05,
  }
  pose["running_threshold"] = 1.8

  cfg.rewards["upright"].weight = 1.5
  cfg.rewards["air_time"].weight = 0.25
  cfg.rewards["air_time"].params["threshold_min"] = 0.1
  cfg.rewards["air_time"].params["command_threshold"] = 0.3
  cfg.rewards["contact_pattern"] = RewardTermCfg(
    func=gait_rewards.contact_pattern,
    weight=0.5,
    params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
  )
  cfg.rewards["gait_symmetry"] = RewardTermCfg(
    func=gait_rewards.gait_symmetry,
    weight=-2.0,
    params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
  )
  cfg.rewards["arm_leg_swing"] = RewardTermCfg(
    func=gait_rewards.arm_leg_swing,
    weight=0.75,
    params={
      "command_name": "twist",
      "std": 0.2,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          "left_shoulder_pitch_joint",
          "right_shoulder_pitch_joint",
          "left_hip_pitch_joint",
          "right_hip_pitch_joint",
        ),
        preserve_order=True,
      ),
    },
  )
  feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)
  cfg.rewards["stance_width"] = RewardTermCfg(
    func=gait_rewards.stance_width,
    weight=0.25,
    params={"target": 0.22, "asset_cfg": feet},
  )
  # Home pose: 0.889 m with 0.5 rad knees; straight legs: 0.912 m.
  cfg.rewards["leg_extension"] = RewardTermCfg(
    func=gait_rewards.leg_extension,
    weight=0.5,
    params={"target_height": 0.9, "command_name": "twist", "asset_cfg": feet},
  )
  cfg.rewards["mechanical_power"] = RewardTermCfg(
    func=gait_rewards.mechanical_power,
    weight=-5.0e-4,
  )

  return cfg


def hu_d03_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create LimX HU_D03 flat terrain velocity configuration."""
  cfg = _hu_d03_base_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
