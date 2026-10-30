"""Convert a Unitree G1 motion CSV (LAFAN1 retargeting dataset format) to the HU_D03.

Joint-space transfer, plain MuJoCo on CPU:
  * Joints with the same name are copied. The sign of each joint is chosen by
    comparing the two robots' joint axes in the world frame at the zero pose.
  * The G1 wrist pitch/yaw go to the HU_D03 wrist pitch / hand yaw; the G1
    wrist roll and the HU_D03 head stay at 0.
  * The elbows get a fixed offset because the zero poses differ (ZERO_OFFSETS).
  * Values are clipped to the HU_D03 joint limits.
  * The root height is scaled by the ratio of pelvis heights, then each frame
    is shifted so the feet do not go through the ground.

The HU_D03 is ~1.6 m tall and the G1 ~1.3 m, so foot placement and hand
positions are only approximate. It is good enough to train a tracking policy;
for exact contacts, retarget from the human data with an IK tool instead.

Input CSV (30 fps): x y z qx qy qz qw + 29 G1 joints.
Output CSV (same fps): x y z qx qy qz qw + 31 HU_D03 joints (model order).

  python scripts/retarget_g1_to_hu_d03.py --input walk1_subject1.csv \
      --output walk1_subject1_hu_d03.csv [--g1-xml path/to/g1.xml]
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hu_d03_mjlab.robot.hu_d03_spec import get_spec  # noqa: E402

G1_JOINTS = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)  # fmt: skip

# HU_D03 joint -> G1 joint, where the names differ.
RENAMED = {
  "left_hand_yaw_joint": "left_wrist_yaw_joint",
  "right_hand_yaw_joint": "right_wrist_yaw_joint",
}
# Zero-pose differences (HU_D03 angle = sign * G1 angle + offset). At zero the
# G1 forearm points forward, the HU_D03 forearm hangs down; found by matching
# the forearm direction of both robots at the G1 zero pose.
ZERO_OFFSETS = {"left_elbow_joint": -1.4984, "right_elbow_joint": -1.4984}
UNMAPPED = {"head_yaw_joint", "head_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint"}

HU_FOOT_POINTS = tuple(
  f"contact_{side}_{p}" for side in ("left", "right") for p in ("heel", "center", "tip")
)


def default_g1_xml() -> Path:
  try:
    from mjlab import MJLAB_SRC_PATH

    return MJLAB_SRC_PATH / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
  except ImportError:
    root = Path(__file__).resolve().parents[1]
    return root / "third_party/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"


def world_axes(model: mujoco.MjModel) -> dict[str, np.ndarray]:
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)
  return {
    model.joint(j).name: data.xaxis[j].copy()
    for j in range(model.njnt)
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
  }


def pelvis_height(model: mujoco.MjModel, foot_sites: tuple[str, ...]) -> float:
  data = mujoco.MjData(model)
  data.qpos[2] = 1.0
  mujoco.mj_kinematics(model, data)
  return 1.0 - min(data.site(s).xpos[2] for s in foot_sites)


def build_mapping(g1: mujoco.MjModel, hu: mujoco.MjModel):
  """Returns (hu_qpos_index, g1_column, sign) triples and a printable report."""
  g1_axes, hu_axes = world_axes(g1), world_axes(hu)
  mapping, report = [], []
  for j in range(1, hu.njnt):
    name = hu.joint(j).name
    if name in UNMAPPED:
      report.append(f"  {name:28s} <- 0")
      continue
    src = RENAMED.get(name, name)
    dot = float(np.dot(g1_axes[src], hu_axes[name]))
    if abs(dot) < 0.7:
      report.append(f"  {name:28s} <- 0 (axes differ, dot={dot:+.2f})")
      continue
    sign = 1.0 if dot > 0 else -1.0
    mapping.append((hu.jnt_qposadr[j], G1_JOINTS.index(src), sign))
    report.append(f"  {name:28s} <- {'+' if sign > 0 else '-'}{src} (dot={dot:+.2f})")
  return mapping, report


def retarget(motion: np.ndarray, g1: mujoco.MjModel, hu: mujoco.MjModel, verbose=True):
  mapping, report = build_mapping(g1, hu)
  if verbose:
    print("joint mapping:\n" + "\n".join(report))

  scale = pelvis_height(hu, ("left_foot", "right_foot")) / pelvis_height(
    g1, ("left_foot", "right_foot")
  )
  lo = hu.jnt_range[1:, 0] * 0.98
  hi = hu.jnt_range[1:, 1] * 0.98

  n = motion.shape[0]
  qpos = np.zeros((n, hu.nq))
  qpos[:, :3] = motion[:, :3] * scale
  qpos[:, 3:7] = motion[:, [6, 3, 4, 5]]  # xyzw -> wxyz
  for adr, col, sign in mapping:
    qpos[:, adr] = sign * motion[:, 7 + col]
  for name, offset in ZERO_OFFSETS.items():
    qpos[:, hu.jnt_qposadr[hu.joint(name).id]] += offset
  qpos[:, 7:] = np.clip(qpos[:, 7:], lo, hi)

  # Ground the feet: global shift so the feet touch the ground in most frames,
  # then lift any frame where a foot point is still below the ground.
  data = mujoco.MjData(hu)
  min_z = np.empty(n)
  for i in range(n):
    data.qpos[:] = qpos[i]
    mujoco.mj_kinematics(hu, data)
    min_z[i] = min(data.body(b).xpos[2] for b in HU_FOOT_POINTS)
  qpos[:, 2] -= np.percentile(min_z, 5)
  min_z -= np.percentile(min_z, 5)
  qpos[:, 2] += np.maximum(-min_z, 0.0)

  if verbose:
    print(f"root height scale {scale:.3f}, frames {n}")
  out = np.concatenate([qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]], axis=1)
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--input", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--g1-xml", default=None)
  args = parser.parse_args()

  g1 = mujoco.MjModel.from_xml_path(str(args.g1_xml or default_g1_xml()))
  hu = get_spec().compile()
  motion = np.loadtxt(args.input, delimiter=",")
  assert motion.shape[1] == 7 + len(G1_JOINTS), motion.shape
  out = retarget(motion, g1, hu)
  np.savetxt(args.output, out, delimiter=",", fmt="%.6f")
  print(f"wrote {args.output} ({out.shape[0]} frames, {out.shape[1] - 7} joints)")


if __name__ == "__main__":
  main()
