"""CPU stand test for the converted HU_D03 model (plain MuJoCo, no GPU, no mjlab).

Adds PD position actuators with the gains from hu_d03_params, puts the robot in
the home pose with its soles on the ground, and holds the pose for a few
seconds. Prints the home base height, any self-contacts at the home pose, and
how far the base drifts.

  python scripts/stand_test.py [--seconds 5] [--viewer]
"""

import argparse
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hu_d03_mjlab.robot.hu_d03_params import ACTUATOR_GROUPS, HOME_JOINT_POS  # noqa: E402
from hu_d03_mjlab.robot.hu_d03_spec import get_spec  # noqa: E402


def match(patterns, name):
  return any(re.fullmatch(p, name) for p in patterns)


def build_model():
  spec = get_spec()
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 1),
  )
  spec.option.timestep = 0.005
  for joint in spec.joints:
    if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    group = next(g for g in ACTUATOR_GROUPS.values() if match(g.joint_names_expr, joint.name))
    joint.armature = group.armature
    act = spec.add_actuator(name=joint.name, target=joint.name,
                            trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=group.stiffness, kv=group.damping)
    act.forcelimited = True
    act.forcerange = (-group.effort_limit, group.effort_limit)
  return spec.compile()


def home_qpos(model):
  qpos = model.qpos0.copy()
  for j in range(1, model.njnt):
    name = model.joint(j).name
    for pattern, value in HOME_JOINT_POS.items():
      if re.fullmatch(pattern, name):
        qpos[model.jnt_qposadr[j]] = value
  # Lift or lower the base so the lowest foot site touches z = 0.
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qpos[2] = 1.0
  mujoco.mj_kinematics(model, data)
  sole = min(data.site("left_foot").xpos[2], data.site("right_foot").xpos[2])
  qpos[2] = 1.0 - sole
  return qpos


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seconds", type=float, default=5.0)
  parser.add_argument("--viewer", action="store_true")
  args = parser.parse_args()

  model = build_model()
  data = mujoco.MjData(model)
  qpos = home_qpos(model)
  data.qpos[:] = qpos
  data.ctrl[:] = qpos[7:]
  mujoco.mj_forward(model, data)

  print(f"home base height: {qpos[2]:.4f} m, total mass {model.body_mass.sum():.2f} kg")
  floor = model.geom("floor").id
  self_contacts = [
    (model.geom(c.geom1).name, model.geom(c.geom2).name)
    for c in data.contact[: data.ncon]
    if floor not in (c.geom1, c.geom2)
  ]
  print(f"self-contacts at home pose: {self_contacts or 'none'}")

  def step():
    mujoco.mj_step(model, data)

  if args.viewer:
    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as v:
      while v.is_running() and data.time < args.seconds:
        step()
        v.sync()
  else:
    while data.time < args.seconds:
      step()

  drift = np.linalg.norm(data.qpos[:2] - qpos[:2])
  up = data.xmat[model.body("base_link").id].reshape(3, 3)[:, 2]
  tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
  joint_err = np.abs(data.qpos[7:] - qpos[7:]).max()
  print(f"after {args.seconds:.1f}s: base z {data.qpos[2]:.4f} m, xy drift {drift:.4f} m, "
        f"tilt {tilt:.2f} deg, max joint error {joint_err:.4f} rad")
  ok = data.qpos[2] > qpos[2] - 0.1 and tilt < 10
  print("STAND TEST", "PASSED" if ok else "FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
