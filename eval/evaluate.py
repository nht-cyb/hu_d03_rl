"""Velocity tracking per checkpoint from the training TensorBoard logs."""
import re, sys, zipfile, json
import numpy as np
from tfevents import read

RUN = "2026-10-28_05-49-07"
ev = read(f"data/hu_d03_rl/logs/rsl_rl/hu_d03_velocity/{RUN}/events.out.tfevents.1793166552.8a568b1f11fa.1084.0")
z = zipfile.ZipFile(sys.argv[1])
ckpts = sorted(int(m.group(1)) for n in z.namelist() if RUN in n and (m := re.search(r"model_(\d+)\.pt$", n)))

def series(tag):
  d = ev[tag]; return np.array([d.get(i, np.nan) for i in range(10000)])
S = {k: series(t) for k, t in {
  "lin_r": "Episode_Reward/track_linear_velocity", "ang_r": "Episode_Reward/track_angular_velocity",
  "err_xy": "Metrics/twist/error_vel_xy", "err_yaw": "Metrics/twist/error_vel_yaw",
  "len": "Train/mean_episode_length", "fell": "Episode_Termination/fell_over",
  "tout": "Episode_Termination/time_out", "rew": "Train/mean_reward",
  "vx_lo": "Curriculum/command_vel/lin_vel_x_min", "vx_hi": "Curriculum/command_vel/lin_vel_x_max",
  "wz_hi": "Curriculum/command_vel/ang_vel_z_max"}.items()}

rows = []
for c in ckpts:
  w = slice(max(0, c - 49), c + 1)          # the 50 iterations leading up to the save
  m = {k: float(np.nanmean(v[w])) for k, v in S.items()}
  L = m["len"]                                # policy steps (max 1000 = 20 s)
  rows.append(dict(ckpt=c, **m,
    lin_q=m["lin_r"] / 2.0 * 1000 / L,        # mean exp-kernel per alive step, 1 = perfect
    ang_q=m["ang_r"] / 2.0 * 1000 / L,
    err_xy_ms=m["err_xy"] * 400 / L,          # metric sums err/400 steps over the episode
    err_yaw_rads=m["err_yaw"] * 400 / L,
    fall_pct=100 * m["fell"] / (m["fell"] + m["tout"])))
json.dump(rows, open("rows.json", "w"))
print(f"{len(ckpts)} checkpoints: {ckpts[:3]} ... {ckpts[-3:]}")
hdr = "ckpt  len   fall%  linRew angRew  linQ  angQ  err_xy  err_yaw  vx range      wz max"
print(hdr)
for r in rows:
  if r["ckpt"] % 500 == 0 or r["ckpt"] in (ckpts[0], ckpts[-1]):
    print(f"{r['ckpt']:5d} {r['len']:5.0f} {r['fall_pct']:6.1f}  {r['lin_r']:.3f}  {r['ang_r']:.3f}  {r['lin_q']:.3f} {r['ang_q']:.3f}  {r['err_xy_ms']:.3f}   {r['err_yaw_rads']:.3f}   [{r['vx_lo']:+.1f},{r['vx_hi']:+.1f}]  {r['wz_hi']:.1f}")
late = [r for r in rows if r["ckpt"] >= 3000]
for key, best in [("lin_r", max), ("ang_r", max), ("err_xy_ms", min), ("err_yaw_rads", min)]:
  b = best(late, key=lambda r: r[key]); print(f"best {key}: ckpt {b['ckpt']} = {b[key]:.3f}")
