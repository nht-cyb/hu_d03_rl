"""Load the LimX HU_D03 MJCF and turn it into a serial-joint model for mjlab.

This module only depends on `mujoco`, so it can be tested on a CPU-only
machine without mjlab installed.

What the conversion does:
  * Removes the parallel ankle (A/B achilles) and waist (A/B) linkages and the 6
    `<connect>` constraints that close them. The policy then commands the
    serial joints `ankle_pitch/roll` and `waist_roll/pitch` directly, like the
    Unitree G1 model in mjlab. The linkage mass is added to the parent body.
  * Removes the floor, light, camera, actuators and sensors from the source XML
    (mjlab adds its own terrain, actuators and sensors).
  * Sets every geom margin to 0 (MuJoCo Warp does not support margins on
    box/mesh pairs, such as the two foot boxes).
  * Names every collision geom (`<body>_collision<i>`, feet are
    `left_foot_collision` / `right_foot_collision`) so mjlab can match them.
  * Adds foot sites at the sole and the IMU sensors mjlab's tasks read.
"""

import os
from pathlib import Path

import mujoco
import numpy as np

_DEFAULT_DESCRIPTION_DIR = (
  Path(__file__).resolve().parents[3]
  / "third_party"
  / "humanoid-description"
  / "HU_D03_description"
)


def description_dir() -> Path:
  """Directory of `HU_D03_description` (override with $HU_D03_DESCRIPTION_DIR)."""
  return Path(os.environ.get("HU_D03_DESCRIPTION_DIR", _DEFAULT_DESCRIPTION_DIR))


def source_xml() -> Path:
  xml = description_dir() / "xml" / "HU_D03_03.xml"
  if not xml.exists():
    raise FileNotFoundError(
      f"{xml} not found. Clone https://github.com/limxdynamics/humanoid-description "
      "and set HU_D03_DESCRIPTION_DIR to its HU_D03_description folder."
    )
  return xml


# Bodies (with their subtrees) that form the closed-loop linkages.
LINKAGE_BODIES = (
  "left_A_achilles_link",
  "left_B_achilles_link",
  "right_A_achilles_link",
  "right_B_achilles_link",
  "waist_A_link",
  "waist_B_link",
)

# Foot sole: the foot box is centred 0.0535 m below the ankle roll frame and is
# 0.01 m half-thick, so the sole is at z = -0.0635.
FOOT_SITE_POS = (0.018, 0.0, -0.0635)
FOOT_GEOMS = {"left_foot": "left_foot_collision", "right_foot": "right_foot_collision"}


def _subtree_mass(body: mujoco.MjsBody) -> float:
  mass = body.mass
  for child in body.bodies:
    mass += _subtree_mass(child)
  return mass


def _add_mass(body: mujoco.MjsBody, extra: float) -> None:
  """Add point-like mass to a body, scaling its inertia by the same ratio."""
  ratio = (body.mass + extra) / body.mass
  body.mass = body.mass + extra
  body.inertia = np.asarray(body.inertia) * ratio


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(source_xml()))

  # Equality constraints, actuators and sensors all reference linkage joints.
  for eq in list(spec.equalities):
    spec.delete(eq)
  for act in list(spec.actuators):
    spec.delete(act)
  for sensor in list(spec.sensors):
    spec.delete(sensor)

  # Floor, light and tracking camera live in the worldbody; mjlab adds terrain.
  for geom in list(spec.worldbody.geoms):
    spec.delete(geom)
  for light in list(spec.worldbody.lights):
    spec.delete(light)
  for cam in list(spec.worldbody.cameras):
    spec.delete(cam)

  # Remove the linkages, moving their mass to the parent body.
  for name in LINKAGE_BODIES:
    body = spec.body(name)
    parent = body.parent
    _add_mass(parent, _subtree_mass(body))
    spec.delete(body)

  # The free joint has a tiny spring/damper in the source XML; drop them.
  root = spec.body("base_link")
  free = root.joints[0]
  free.damping = np.zeros_like(free.damping)
  free.stiffness = np.zeros_like(free.stiffness)

  # Name collision geoms so mjlab's regex-based configs can find them.
  for body in spec.bodies:
    i = 0
    for geom in body.geoms:
      # The XML default gives every geom margin=0.001. MuJoCo Warp rejects a
      # non-zero margin on box/mesh pairs (e.g. foot-foot) with MULTICCD.
      geom.margin = 0.0
      if geom.contype == 0 and geom.conaffinity == 0:
        continue  # Visual mesh.
      if geom.name in FOOT_GEOMS:
        geom.name = FOOT_GEOMS[geom.name]
      elif not geom.name:
        geom.name = f"{body.name}_collision{i}"
        i += 1

  for side in ("left", "right"):
    spec.body(f"{side}_ankle_roll_link").add_site(
      name=f"{side}_foot", pos=FOOT_SITE_POS, size=(0.01, 0.01, 0.01), group=5
    )

  # Sensors read by mjlab's velocity and tracking tasks.
  spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_lin_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_upvector", type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
                  objtype=mujoco.mjtObj.mjOBJ_BODY, objname="world",
                  reftype=mujoco.mjtObj.mjOBJ_SITE, refname="imu")
  spec.add_sensor(name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
                  objtype=mujoco.mjtObj.mjOBJ_BODY, objname="base_link")

  return spec
