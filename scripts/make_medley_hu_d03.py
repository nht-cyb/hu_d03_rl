"""Join several HU_D03 motion CSVs into one continuous clip (a "medley").

One tracking policy trained on the medley learns all of its dances: the policy
observes the reference joint positions and velocities, so it tracks whichever
part of the clip it is given, and mjlab's adaptive sampling spends more time on
the parts it fails at.

Each clip is turned about the vertical axis and shifted so it starts where the
previous one ended (same heading and floor position), then a short blend is
inserted: joints and root height are interpolated with a smoothstep, the root
orientation is slerped. Plain numpy, CPU.

Input/output CSV (30 fps): x y z qx qy qz qw + HU_D03 joints (model order).
A <output>.segments.json next to the output lists where each clip starts and ends.

  python scripts/make_medley_hu_d03.py --inputs a_hu_d03.csv b_hu_d03.csv \
      --output medley_hu_d03.csv [--blend-seconds 1.0] [--fps 30]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def quat_mul(a, b):
  """Hamilton product of xyzw quaternions (broadcasts)."""
  ax, ay, az, aw = np.moveaxis(a, -1, 0)
  bx, by, bz, bw = np.moveaxis(b, -1, 0)
  return np.stack([
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ], axis=-1)


def yaw(q):
  x, y, z, w = q
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def slerp(q0, q1, t):
  d = np.dot(q0, q1)
  if d < 0:
    q1, d = -q1, -d
  if d > 0.9995:
    q = q0 + t * (q1 - q0)
    return q / np.linalg.norm(q)
  th = np.arccos(d)
  return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def align(clip, prev):
  """Turn and shift clip so its first frame has prev's heading and floor position."""
  dyaw = yaw(prev[3:7]) - yaw(clip[0, 3:7])
  c, s = np.cos(dyaw), np.sin(dyaw)
  out = clip.copy()
  xy = clip[:, :2] - clip[0, :2]
  out[:, 0] = c * xy[:, 0] - s * xy[:, 1] + prev[0]
  out[:, 1] = s * xy[:, 0] + c * xy[:, 1] + prev[1]
  qz = np.array([0.0, 0.0, np.sin(dyaw / 2), np.cos(dyaw / 2)])
  out[:, 3:7] = quat_mul(qz, clip[:, 3:7])
  return out


def blend(a, b, n):
  """n frames going from frame a to frame b (both excluded)."""
  frames = []
  for k in range(1, n + 1):
    t = k / (n + 1)
    t = t * t * (3 - 2 * t)  # Smoothstep: zero velocity change at both ends of the blend.
    f = a + t * (b - a)
    f[3:7] = slerp(a[3:7], b[3:7], t)
    frames.append(f)
  return np.array(frames).reshape(n, a.shape[0])


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--inputs", nargs="+", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--blend-seconds", type=float, default=1.0)
  parser.add_argument("--fps", type=float, default=30.0)
  args = parser.parse_args()

  n_blend = round(args.blend_seconds * args.fps)
  out = np.loadtxt(args.inputs[0], delimiter=",", ndmin=2)
  segments = [{"clip": Path(args.inputs[0]).stem, "start_s": 0.0, "end_s": len(out) / args.fps}]
  for path in args.inputs[1:]:
    clip = np.loadtxt(path, delimiter=",", ndmin=2)
    assert clip.shape[1] == out.shape[1], (path, clip.shape)
    clip = align(clip, out[-1])
    out = np.concatenate([out, blend(out[-1], clip[0], n_blend), clip])
    segments.append({"clip": Path(path).stem, "start_s": (len(out) - len(clip)) / args.fps,
                     "end_s": len(out) / args.fps})

  np.savetxt(args.output, out, delimiter=",", fmt="%.6f")
  Path(args.output).with_suffix(".segments.json").write_text(json.dumps(segments, indent=1))
  for s in segments:
    print(f"  {s['start_s']:7.1f} - {s['end_s']:7.1f} s  {s['clip']}")
  print(f"wrote {args.output} ({len(out)} frames, {len(out) / args.fps:.1f} s, "
        f"{len(segments)} clips, {args.blend_seconds:g} s blends)")


if __name__ == "__main__":
  main()
