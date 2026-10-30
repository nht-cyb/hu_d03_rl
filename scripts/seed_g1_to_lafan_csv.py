"""Convert a BONES-SEED G1 CSV to the LAFAN1 G1 CSV format retarget_g1_to_hu_d03.py reads.

BONES-SEED (huggingface.co/datasets/bones-studio/seed, g1/csv/<date>/<name>.csv), 120 fps:
  header row; Frame, root_translateX/Y/Z (cm), root_rotateX/Y/Z (extrinsic XYZ
  Euler, degrees), <joint>_dof (degrees, G1 MJCF joint names).
LAFAN1 G1 format, 30 fps by default:
  no header; x y z (m), qx qy qz qw, 29 joints (radians) in G1_JOINTS order.

  python scripts/seed_g1_to_lafan_csv.py --input ballet.csv --output ballet_g1.csv \
      [--input-fps 120] [--output-fps 30] [--start-s 0] [--end-s 30]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retarget_g1_to_hu_d03 import G1_JOINTS  # noqa: E402


def euler_xyz_extrinsic_to_quat_xyzw(euler_deg: np.ndarray) -> np.ndarray:
  """Extrinsic X, then Y, then Z rotations (R = Rz Ry Rx) as xyzw quaternions."""
  half = np.deg2rad(euler_deg) / 2
  cx, cy, cz = np.cos(half).T
  sx, sy, sz = np.sin(half).T
  # q = qz * qy * qx
  w = cz * cy * cx + sz * sy * sx
  x = cz * cy * sx - sz * sy * cx
  y = cz * sy * cx + sz * cy * sx
  z = sz * cy * cx - cz * sy * sx
  return np.stack([x, y, z, w], axis=1)


def convert(path: str, input_fps: int, output_fps: int, start_s: float, end_s: float | None):
  with open(path) as f:
    header = [h.strip() for h in f.readline().split(",")]
  data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
  col = {name: i for i, name in enumerate(header)}

  pos = data[:, [col[f"root_translate{a}"] for a in "XYZ"]] / 100.0
  quat = euler_xyz_extrinsic_to_quat_xyzw(data[:, [col[f"root_rotate{a}"] for a in "XYZ"]])

  # Match joints by name; fall back to column order if the names differ.
  names = [f"{j}_dof" for j in G1_JOINTS]
  if all(n in col for n in names):
    joints = data[:, [col[n] for n in names]]
  else:
    joints = data[:, 7:]
    assert joints.shape[1] == len(G1_JOINTS), (
      f"expected {len(G1_JOINTS)} joint columns, got {joints.shape[1]}: {header[7:]}"
    )
    print("joint names differ from the G1 MJCF; using column order:", header[7:])
  joints = np.deg2rad(joints)

  if input_fps % output_fps:
    raise ValueError(f"input fps {input_fps} is not a multiple of output fps {output_fps}")
  first = int(round(start_s * input_fps))
  last = None if end_s is None else int(round(end_s * input_fps))
  step = input_fps // output_fps
  return np.concatenate([pos, quat, joints], axis=1)[first:last:step]


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--input", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--input-fps", type=int, default=120)
  parser.add_argument("--output-fps", type=int, default=30)
  parser.add_argument("--start-s", type=float, default=0.0)
  parser.add_argument("--end-s", type=float, default=None)
  args = parser.parse_args()

  out = convert(args.input, args.input_fps, args.output_fps, args.start_s, args.end_s)
  np.savetxt(args.output, out, delimiter=",", fmt="%.6f")
  print(f"wrote {args.output} ({out.shape[0]} frames at {args.output_fps} fps, "
        f"{out.shape[0] / args.output_fps:.1f} s)")


if __name__ == "__main__":
  main()
