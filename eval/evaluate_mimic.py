"""Motion tracking per checkpoint from the training TensorBoard logs of the mimic runs.

  python eval/evaluate_mimic.py ballet=mimic_ballet_full_run.zip medley=mimic_medley6_full_run.zip

Each zip is a full run from the training-results release. Writes eval/mimic_rows.json and
eval/mimic_tracking.png.
"""
import json, re, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tfevents import read

OUT = Path(__file__).resolve().parent
N_JOINTS = 31                                 # HU_D03 actuated joints
MAX_LEN = 500                                 # episode steps (10 s at 50 Hz)
TAGS = {
  "anchor_pos": "Metrics/motion/error_anchor_pos", "anchor_rot": "Metrics/motion/error_anchor_rot",
  "body_pos": "Metrics/motion/error_body_pos", "body_rot": "Metrics/motion/error_body_rot",
  "joint_pos": "Metrics/motion/error_joint_pos", "len": "Train/mean_episode_length",
  "rew": "Train/mean_reward", "tout": "Episode_Termination/time_out",
  "t_anchor_pos": "Episode_Termination/anchor_pos", "t_anchor_ori": "Episode_Termination/anchor_ori",
  "t_ee": "Episode_Termination/ee_body_pos"}

def load(zip_path):
  z = zipfile.ZipFile(zip_path)
  ev_name = next(n for n in z.namelist() if "events.out.tfevents" in n)
  with tempfile.NamedTemporaryFile() as f:
    f.write(z.read(ev_name)); f.flush(); ev = read(f.name)
  ckpts = sorted({int(m.group(1)) for n in z.namelist() if (m := re.search(r"logs/.*model_(\d+)\.pt$", n))})
  series = {k: np.array([ev[t].get(i, np.nan) for i in range(10000)]) for k, t in TAGS.items()}
  rows = []
  for c in ckpts:
    w = slice(max(0, c - 49), c + 1)          # the 50 iterations leading up to the save
    m = {k: float(np.nanmean(v[w])) for k, v in series.items()}
    early = m["t_anchor_pos"] + m["t_anchor_ori"] + m["t_ee"]
    rows.append(dict(ckpt=c, **m,
      body_pos_cm=100 * m["body_pos"],           # mean over tracked bodies, relative to the torso
      body_rot_deg=np.degrees(m["body_rot"]),
      anchor_pos_cm=100 * m["anchor_pos"],       # torso position in the world (global drift)
      joint_rms_deg=np.degrees(m["joint_pos"] / np.sqrt(N_JOINTS)),
      len_pct=100 * m["len"] / MAX_LEN,
      early_pct=100 * early / (early + m["tout"])))
  return rows

runs = {name: load(path) for name, path in (a.split("=", 1) for a in sys.argv[1:])}
json.dump(runs, open(OUT / "mimic_rows.json", "w"))

PANELS = [("body_pos_cm", "Body position error (cm)"), ("body_rot_deg", "Body orientation error (deg)"),
          ("joint_rms_deg", "Joint angle error, RMS per joint (deg)"), ("anchor_pos_cm", "Torso position drift (cm)"),
          ("len_pct", "Episode length (% of 10 s)"), ("early_pct", "Episodes ended early (%)")]
COLORS = ["#2a78d6", "#eb6834"]
fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6))
for ax, (key, title) in zip(axes.flat, PANELS):
  for (name, rows), color in zip(runs.items(), COLORS):
    ax.plot([r["ckpt"] for r in rows], [r[key] for r in rows], color=color, lw=2, label=name)
  ax.set_title(title, fontsize=10.5, loc="left")
  if key not in ("len_pct", "early_pct"):     # the untrained checkpoint 0 would flatten the rest
    ax.set_ylim(0, 1.25 * max(r[key] for rows in runs.values() for r in rows if r["ckpt"] >= 1000))
  else: ax.set_ylim(0, 105)
  ax.grid(color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
  for s in ("top", "right"): ax.spines[s].set_visible(False)
  for s in ("left", "bottom"): ax.spines[s].set_color("#8a8984")
  ax.tick_params(colors="#55544f", labelsize=9)
for ax in axes[1]: ax.set_xlabel("checkpoint (iteration)", fontsize=9.5, color="#55544f")
axes[0, 0].legend(frameon=False, fontsize=9.5)
fig.text(0.01, 0.005, "Error y-axes stop at 1.25× the highest value after iteration 1000; early values can be off the top.",
         fontsize=8.5, color="#55544f")
fig.suptitle("HU_D03 mimic — motion tracking per checkpoint (training-time averages, 4096 envs)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "mimic_tracking.png", dpi=110)

hdr = f"{'run':8s} {'ckpt':>5s} {'len%':>5s} {'early%':>6s} {'body cm':>7s} {'rot deg':>7s} {'joint deg':>9s} {'drift cm':>8s}"
print(hdr)
for name, rows in runs.items():
  for r in rows:
    if r["ckpt"] % 2500 == 0 or r is rows[-1]:
      print(f"{name:8s} {r['ckpt']:5d} {r['len_pct']:5.1f} {r['early_pct']:6.1f} {r['body_pos_cm']:7.1f} "
            f"{r['body_rot_deg']:7.1f} {r['joint_rms_deg']:9.1f} {r['anchor_pos_cm']:8.1f}")
