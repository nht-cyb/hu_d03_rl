"""HU_D03 actuator and pose parameters (plain Python, no mjlab import).

Armatures are the reflected rotor inertias from the LimX SRDF
(`rotor mass * gear_ratio**2`), which match the `armature` values in the MJCF.
Effort limits are the `ctrlrange` of the MJCF motors.

LimX does not publish the PD gains used on the real robot, so stiffness and
damping follow mjlab's G1 rule: a second-order system with natural frequency
NATURAL_FREQ_HZ and damping ratio DAMPING_RATIO for the joint's armature.
Replace them with the real gains if you get them from LimX.

The ankle and waist roll/pitch joints are driven by two motors through a
parallel linkage. As for the G1, we assume a nominal 1:1 linkage, so these
joints get the armature and effort of both motors.
"""

import math
from dataclasses import dataclass

NATURAL_FREQ_HZ = 10.0
DAMPING_RATIO = 2.0

ARMATURE_LEG = 0.000244114 * 25.0**2  # 0.1526, hips and knees.
ARMATURE_36 = 0.000073217 * 36.0**2  # 0.0949, ankle/waist motors.
ARMATURE_ARM = 0.000073217 * 25.0**2  # 0.0458, shoulders and elbows.
ARMATURE_SMALL = 0.000017 * 25.0**2  # 0.0106, wrists, hands and head.


@dataclass(frozen=True)
class ActuatorGroup:
  joint_names_expr: tuple[str, ...]
  armature: float
  effort_limit: float

  @property
  def stiffness(self) -> float:
    w = 2.0 * math.pi * NATURAL_FREQ_HZ
    return self.armature * w**2

  @property
  def damping(self) -> float:
    w = 2.0 * math.pi * NATURAL_FREQ_HZ
    return 2.0 * DAMPING_RATIO * self.armature * w


ACTUATOR_GROUPS = {
  "leg": ActuatorGroup(
    (".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"),
    ARMATURE_LEG,
    120.0,
  ),
  "ankle": ActuatorGroup(
    (".*_ankle_pitch_joint", ".*_ankle_roll_joint"), 2 * ARMATURE_36, 2 * 45.0
  ),
  "waist_yaw": ActuatorGroup(("waist_yaw_joint",), ARMATURE_36, 45.0),
  "waist": ActuatorGroup(
    ("waist_roll_joint", "waist_pitch_joint"), 2 * ARMATURE_36, 2 * 45.0
  ),
  "arm": ActuatorGroup(
    (
      ".*_shoulder_pitch_joint",
      ".*_shoulder_roll_joint",
      ".*_shoulder_yaw_joint",
      ".*_elbow_joint",
    ),
    ARMATURE_ARM,
    30.0,
  ),
  "small": ActuatorGroup(
    (".*_wrist_yaw_joint", ".*_wrist_pitch_joint", ".*_hand_yaw_joint", "head_.*"),
    ARMATURE_SMALL,
    18.0,
  ),
}

# Standing pose with slightly bent knees. Signs: negative hip pitch swings the
# thigh forward, positive knee flexes, negative elbow flexes. The base height
# puts the soles on the ground (checked by scripts/stand_test.py).
HOME_BASE_HEIGHT = 0.889
HOME_JOINT_POS = {
  ".*_hip_pitch_joint": -0.25,
  ".*_knee_joint": 0.5,
  ".*_ankle_pitch_joint": -0.25,
  "left_shoulder_roll_joint": 0.15,
  "right_shoulder_roll_joint": -0.15,
  ".*_elbow_joint": -0.3,
}
