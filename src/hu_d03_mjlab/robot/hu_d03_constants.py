"""LimX HU_D03 robot config for mjlab."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from hu_d03_mjlab.robot.hu_d03_params import (
  ACTUATOR_GROUPS,
  HOME_BASE_HEIGHT,
  HOME_JOINT_POS,
)
from hu_d03_mjlab.robot.hu_d03_spec import get_spec

##
# Actuator config.
##

HU_D03_ACTUATORS = tuple(
  BuiltinPositionActuatorCfg(
    target_names_expr=g.joint_names_expr,
    stiffness=g.stiffness,
    damping=g.damping,
    effort_limit=g.effort_limit,
    armature=g.armature,
  )
  for g in ACTUATOR_GROUPS.values()
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, HOME_BASE_HEIGHT),
  joint_pos=HOME_JOINT_POS,
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_GEOMS = r"^(left|right)_foot_collision$"

# All collisions, including self collisions. Feet get condim=3, the rest condim=1.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  contype=1,
  conaffinity=1,
  condim={FOOT_GEOMS: 3, ".*_collision.*": 1},
  priority={FOOT_GEOMS: 1, ".*": 0},
  friction={FOOT_GEOMS: (0.6,)},
)

##
# Final config.
##

HU_D03_ARTICULATION = EntityArticulationInfoCfg(
  actuators=HU_D03_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_hu_d03_robot_cfg() -> EntityCfg:
  """Get a fresh HU_D03 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=HU_D03_ARTICULATION,
  )


# With the 10 Hz gains, 0.25 * effort / stiffness (the G1 rule) gives only
# 0.03-0.05 rad per unit action on the legs, because the HU_D03's reflected
# inertias are large. Use a flat 0.25 rad instead.
HU_D03_ACTION_SCALE: dict[str, float] = {".*": 0.25}
