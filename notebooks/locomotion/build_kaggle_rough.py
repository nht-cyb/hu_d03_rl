"""Generate notebooks/locomotion/kaggle_hu_d03_rough.ipynb.

A Kaggle-only notebook that trains Mjlab-Velocity-Rough-LimX-HU-D03, warm-started
from the flat locomotion policy. Like the flat notebook it runs unattended with
"Save Version -> Save & Run All (Commit)". Re-run after editing src/ or scripts/:

  python notebooks/locomotion/build_kaggle_rough.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_common import (  # noqa: E402
  DESCRIPTION_COMMIT,
  MJLAB_COMMIT,
  PACKAGE_FILES,
  ROOT,
  Notebook,
)

OUT = ROOT / "notebooks" / "locomotion" / "kaggle_hu_d03_rough.ipynb"
CODE = "/kaggle/temp/hu_d03_code"
# The commit of nht-cyb/hu_d03_rl that holds the flat policy from 2026-10-28.
FLAT_POLICY_COMMIT = "1a7b1e6"

nb = Notebook()
md, code = nb.md, nb.code

md("""
# LimX HU_D03 rough terrain locomotion on Kaggle (mjlab), warm-started from the flat policy

Trains `Mjlab-Velocity-Rough-LimX-HU-D03`: the HU_D03 walks and runs on stairs, slopes, rough ground, waves and low
obstacles, with a human-like gait. It starts from the trained flat policy, `hu_d03_velocity_flat.pt`, instead of from
scratch.

### Before running

1. **Settings → Accelerator → GPU T4 x2.** Not P100: it is older than Turing and mjlab does not support it.
2. **Settings → Internet → On** (needs a phone-verified Kaggle account).

### How to run

* **Try it interactively first:** run the cells down to *5. Sanity checks*. The last video there is the warm-start
  policy on the new terrain, before any training: it should walk like the flat policy and stumble on obstacles.
* **Train with Save Version → Save & Run All (Commit).** That runs the whole notebook in the background (up to 12 h)
  and keeps everything in `/kaggle/working`. Training stops itself after `TIME_BUDGET_HOURS`.
* **See the results in section 8:** one overview video of every terrain, a measured results table, then each clip.

### Continue training in a new version

Open the notebook, *Add Input → Your Work → Notebooks*, and add **this notebook's previous version**. The settings
cell finds its newest checkpoint and resumes from it, with the terrain and speed curricula where they were.

### How the warm start works

A plain resume cannot load the flat checkpoint, for three reasons that `scripts/widen_flat_checkpoint.py` handles:

| Problem | What the script does |
|---|---|
| The rough task adds 187 height-scan inputs (actor 105 → 292, critic 117 → 304) | Adds them with zero weights, so the policy starts out acting exactly like the flat one and learns to use the scan |
| The observation normalizer has seen 1e9 samples and would barely adapt | Seeds the scan's statistics and lowers the sample count |
| The saved step counter (240,000) would start the speed curriculum at its last stage | Resets it, and the iteration count, to 0 |

It also clears Adam's state, whose shapes no longer fit, and raises the action noise to `WARM_START_STD`.

### What makes the gait human-like

There is no motion capture here: the gait comes from these rewards on top of the flat task's.

| Term | What it asks for |
|---|---|
| `contact_pattern` | Both feet down when standing, one foot when walking, at most one when running (allows a flight phase) |
| `air_time` | Long swings instead of short shuffling steps (it was off in the flat task) |
| `gait_symmetry` | Equal swing and stance times on both legs: no limp |
| `arm_leg_swing` | Each arm swings with the opposite leg |
| `leg_extension` | Walk tall, not crouched; measured from the lower foot, so it holds on slopes and stairs |
| `stance_width` | Feet about 22 cm apart |
| `mechanical_power` | Less wasted energy |
| `pose` | Tighter hips and shoulder roll (no flared arms), looser shoulder pitch and waist yaw |

The weights are a reasoned first guess, checked only for their relative size. If the result looks wrong, see
*Tuning the gait* at the end of the notebook.

Commands also change: mostly forward, little strafing, and up to 3 m/s (the flat task reached 2 m/s).
""")

md("## 1. Settings")
code(f"""
import os, shutil, signal, subprocess, urllib.request
from pathlib import Path

WORK = Path("/kaggle/working/hu_d03_rl_rough")   # Saved with each version.
CODE = Path("{CODE}")    # Installed code, re-created each run.
WORK.mkdir(parents=True, exist_ok=True)
CODE.mkdir(parents=True, exist_ok=True)

NUM_ENVS = 4096            # Use 2048 if the GPU runs out of memory.
MAX_ITERATIONS = 10_000    # Upper bound; the time budget may stop training first.
TIME_BUDGET_HOURS = 10.5   # Kaggle stops a version at 12 h; leave time for the videos and zip.
USE_TWO_GPUS = False       # True splits envs over both T4s (faster, but less tested).

WARM_START_FROM = ""       # "" picks automatically: a flat run under /kaggle/input, else the policy on GitHub.
                           # To choose, give a model_*.pt (or a pattern), e.g. "**/hu-d03-flat/**/model_9999.pt".
WARM_START_STD = 0.3       # Raise the policy's action noise to at least this (the flat policy ends at 0.15-0.6).
WARM_START_LR = 3.0e-4     # Starting learning rate; a fresh run's 1e-3 would disturb a policy that already walks.
FLAT_POLICY_URL = "https://github.com/nht-cyb/hu_d03_rl/raw/{FLAT_POLICY_COMMIT}/results/policies/hu_d03_velocity_flat.pt"

# Code versions. These commits were tested with this notebook; "main" gets the latest.
MJLAB_REF = "{MJLAB_COMMIT}"
DESCRIPTION_REF = "{DESCRIPTION_COMMIT}"

TASK = "Mjlab-Velocity-Rough-LimX-HU-D03"
EXPERIMENT = "hu_d03_velocity_rough"
LOG_ROOT = WORK / "logs" / "rsl_rl" / EXPERIMENT
WARM_RUN = "0000_warm_start_flat"   # Sorts before the dated runs, so "newest run" never picks it.

os.environ["WANDB_MODE"] = "disabled"
os.environ["MUJOCO_GL"] = "egl"
os.environ["MJLAB_WARP_QUIET"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"  # Stream training logs into the notebook.
os.environ["PYTHONFAULTHANDLER"] = "1"  # Print a Python traceback on a crash.
os.environ["HU_D03_DESCRIPTION_DIR"] = str(CODE / "humanoid-description" / "HU_D03_description")

def iteration(ckpt):
  return int(ckpt.stem.split("_")[1])

# Resume: newest checkpoint of an earlier version added under "Input".
def newest_input_run():
  ckpts = [p for p in Path("/kaggle/input").glob(f"**/logs/rsl_rl/{{EXPERIMENT}}/*/model_*.pt")
           if p.parent.name != WARM_RUN]
  if not ckpts:
    return None
  ckpts.sort(key=lambda p: (p.parent.name, iteration(p)))
  return ckpts[-1].parent

# Warm start: the flat policy to begin from when there is nothing to resume.
def flat_checkpoint():
  root = Path("/kaggle/input")
  if WARM_START_FROM:
    pattern = str(Path(WARM_START_FROM).relative_to(root)) if WARM_START_FROM.startswith("/") else WARM_START_FROM
    found = sorted(root.glob(pattern), key=iteration)
    if not found:
      raise SystemExit(f"WARM_START_FROM matches nothing under {{root}}.")
    return found[-1]
  found = sorted(root.glob("**/logs/rsl_rl/hu_d03_velocity/*/model_*.pt"), key=lambda p: (p.parent.name, iteration(p)))
  if found:
    return found[-1]
  dest = WORK / "hu_d03_velocity_flat.pt"
  if not dest.exists():
    print("downloading", FLAT_POLICY_URL)
    urllib.request.urlretrieve(FLAT_POLICY_URL, dest)
  return dest

RESUME_RUN = newest_input_run()
FLAT_CHECKPOINT = None
if RESUME_RUN is not None:
  dest = LOG_ROOT / RESUME_RUN.name
  if not dest.exists():
    shutil.copytree(RESUME_RUN, dest)
  print("will resume from", dest, "| newest checkpoint:", max(dest.glob("model_*.pt"), key=iteration).name)
else:
  FLAT_CHECKPOINT = flat_checkpoint()
  print("will warm-start from", FLAT_CHECKPOINT)
""")

md("## 2. Check the GPU")
code("""
out = subprocess.run(
  ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
  capture_output=True, text=True,
)
if out.returncode != 0:
  raise SystemExit("No GPU. Set Settings → Accelerator → GPU T4 x2.")
print(out.stdout)
name, cap, driver, _ = [s.strip() for s in out.stdout.splitlines()[0].split(",")]
if float(cap) < 7.5:
  raise SystemExit(f"{name} (compute {cap}) is too old for mjlab. Pick GPU T4 x2, not P100.")
CUDA_EXTRA = "cu130" if int(driver.split(".")[0]) >= 580 else "cu126"
NUM_GPUS = len(out.stdout.splitlines())
print("mjlab extra:", CUDA_EXTRA, "| GPUs:", NUM_GPUS)
""")

md("## 3. Install mjlab and the HU_D03 model (about 3–5 minutes)")
code("""
import threading, time

def sh(cmd, cwd=None, max_seconds=None, check=True, log_file=None, on_line=None, on_tick=None):
  \"\"\"Run a shell command and stream its output into the notebook.

  log_file: write the output there instead of the notebook (the last lines are
  shown if the command fails). on_line(line) sees every output line;
  on_tick(elapsed_s) is called every few seconds from the notebook thread.
  After max_seconds the command gets Ctrl-C (then a kill) and counts as done.
  \"\"\"
  print("$", cmd, flush=True)
  log = open(log_file, "a") if log_file else None
  proc = subprocess.Popen(cmd, shell=True, cwd=cwd, text=True, bufsize=1,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          start_new_session=True)
  sh.last_output = []

  def pump_output():
    for line in proc.stdout:
      if log:
        log.write(line)
      else:
        print(line, end="", flush=True)
      if on_line:
        on_line(line)
      sh.last_output.append(line)
      del sh.last_output[:-200]

  pump = threading.Thread(target=pump_output)
  pump.start()
  start, timed_out = time.time(), False
  while proc.poll() is None:
    if on_tick:
      on_tick(time.time() - start)
    if max_seconds is not None and time.time() - start > max_seconds:
      timed_out = True
      print(f"\\n[time budget reached after {max_seconds / 3600:.1f} h: stopping]", flush=True)
      os.killpg(proc.pid, signal.SIGINT)
      try:
        proc.wait(timeout=120)
      except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
      break
    time.sleep(10 if on_tick else 0.5)
  pump.join(timeout=30)
  if on_tick:
    on_tick(time.time() - start)
  if log:
    log.close()
  if check and proc.returncode != 0 and not timed_out:
    if log:
      print("".join(sh.last_output[-60:]))
    raise SystemExit(f"command failed with exit code {proc.returncode}")
  return proc.returncode

def fetch(repo, ref, dest):
  if not (dest / ".git").exists():
    sh(f"git init -q {dest} && git -C {dest} remote add origin {repo}")
  sh(f"git -C {dest} fetch -q --depth 1 origin {ref} && git -C {dest} checkout -q FETCH_HEAD")
  sh(f"git -C {dest} log -1 --format='%h %cd %s' --date=short")

fetch("https://github.com/mujocolab/mjlab.git", MJLAB_REF, CODE / "mjlab")
fetch("https://github.com/limxdynamics/humanoid-description.git", DESCRIPTION_REF, CODE / "humanoid-description")

sh("pip install -q uv")
sh(f"uv pip install --system -q -e '{CODE / 'mjlab'}[{CUDA_EXTRA}]'")
""")

md("""
### Write the HU_D03 package

Generated from the `hu-d03-rl` folder (`notebooks/locomotion/build_kaggle_rough.py`). You can also edit the cells here:
the terrains and reward weights are in `velocity_env_cfgs.py`, the gait rewards in `gait_rewards.py`.
""")
nb.embed(
  PACKAGE_FILES
  + [
    "scripts/stand_test.py",
    "scripts/record_video.py",
    "scripts/widen_flat_checkpoint.py",
    "scripts/make_terrain_overview.py",
  ],
  f"{CODE}/hu-d03-rl",
)
code("""sh(f"uv pip install --system -q -e {CODE / 'hu-d03-rl'}")""")

md("""
## 4. Prepare the warm start

Widens the flat checkpoint for the rough task. See *How the warm start works* above.
""")
code("""
SCRIPTS = CODE / "hu-d03-rl" / "scripts"
WARM_CHECKPOINT = LOG_ROOT / WARM_RUN / "model_0.pt"
if RESUME_RUN is None:
  sh(f"python {SCRIPTS / 'widen_flat_checkpoint.py'} {FLAT_CHECKPOINT} {WARM_CHECKPOINT} "
     f"--min-std {WARM_START_STD} --learning-rate {WARM_START_LR}")
""")

md("""
## 5. Sanity checks

* The stand test holds the home pose in plain MuJoCo (CPU).
* `list-envs` must show the `LimX-HU-D03` tasks. If it does not, look for a
  `[WARN] Failed to load task package 'hu_d03'` traceback.
* The first video runs the full environment on the GPU with zero actions: the robot should stand, then settle on the terrain.
* The second video is the warm-start policy on rough terrain, before training.
  If the error mentions EGL, the cell switches to OSMesa and tries again. Video problems never stop the run.
""")
code("""
sh(f"python {SCRIPTS / 'stand_test.py'} --seconds 10")
sh("list-envs | grep -i hu-d03")
# The values training will really use: 300 and 30. Different numbers mean old code is still installed.
sh('''python -c "from hu_d03_mjlab.tasks.velocity_env_cfgs import hu_d03_rough_env_cfg as f; c = f(); print('ccd_iterations', c.sim.mujoco.ccd_iterations, 'nconmax', c.sim.nconmax)"''')
""")
code("""
from IPython.display import Video, display

def video(out, *extra, show=True):
  \"\"\"Record a video; returns None instead of failing the notebook.\"\"\"
  cmd = f"python {SCRIPTS / 'record_video.py'} {TASK} --output-file {out} " + " ".join(extra)
  if sh(cmd, cwd=WORK, check=False) != 0:
    egl_error = "EGL" in "".join(sh.last_output[-50:])
    if not egl_error:
      print("Could not record", out, "(see the error above)")
      return None
    # OSMesa only for this command: it must not leak into training.
    print("EGL rendering failed; retrying with OSMesa.")
    subprocess.run("apt-get -qq install -y libosmesa6 > /dev/null", shell=True)
    if sh("MUJOCO_GL=osmesa " + cmd, cwd=WORK, check=False) != 0:
      print("Could not record", out, "(see the error above)")
      return None
  if show:
    display(Video(str(WORK / out), embed=True, width=640))
  return WORK / out

video("videos/zero_action.mp4", "--agent zero", "--seconds 5")
if RESUME_RUN is None:
  video("videos/before_training.mp4", f"--checkpoint-file {WARM_CHECKPOINT}", "--forward-speed 1.0", "--seconds 10")
""")

md("""
## 6. Train

Logs, checkpoints (every 50 iterations) and ONNX exports go to `/kaggle/working/hu_d03_rl_rough/logs/rsl_rl/hu_d03_velocity_rough/<date_time>/`.
Training is stopped after `TIME_BUDGET_HOURS`; the newest checkpoint is kept.

The cell shows one progress line, updated every 10 s. The percentage is of whichever limit comes first,
`MAX_ITERATIONS` or `TIME_BUDGET_HOURS`. `terrain` is the mean difficulty level, from 0 up to 9; it should climb steadily.
The full rsl_rl output goes to `hu_d03_rl_rough/train.log`
(`!tail -n 60 /kaggle/working/hu_d03_rl_rough/train.log` shows the latest iteration in detail).
""")
code("""
cmd = (f"MUJOCO_GL=egl train {TASK} --env.scene.num-envs {NUM_ENVS} "
       f"--agent.max-iterations {MAX_ITERATIONS} --agent.logger tensorboard")
if USE_TWO_GPUS and NUM_GPUS >= 2:
  cmd += ' --gpu-ids "[0, 1]"'
if RESUME_RUN is not None:
  cmd += f" --agent.resume True --agent.load-run {RESUME_RUN.name}"
else:
  cmd += f" --agent.resume True --agent.load-run {WARM_RUN} --agent.load-checkpoint model_0.pt"
import html, re
from datetime import timedelta
import IPython.display as ipd

class TrainProgress:
  \"\"\"Parses rsl_rl's per-iteration log and shows one progress line.\"\"\"
  ANSI = re.compile(r"\\x1b\\[[0-9;]*m")
  FIELDS = {
    "iteration": re.compile(r"Learning iteration (\\d+)/(\\d+)"),
    "reward": re.compile(r"Mean reward:\\s+(-?[\\d.]+)"),
    "length": re.compile(r"Mean episode length:\\s+([\\d.]+)"),
    "fps": re.compile(r"Steps per second:\\s+([\\d.]+)"),
    "terrain": re.compile(r"terrain_levels/mean:\\s+([\\d.]+)"),
  }

  def __init__(self, budget_s):
    self.budget_s = budget_s
    self.values = {}
    self.handle = ipd.display(ipd.HTML("<pre>starting training...</pre>"), display_id=True)

  def on_line(self, line):
    line = self.ANSI.sub("", line)
    for key, pattern in self.FIELDS.items():
      m = pattern.search(line)
      if m:
        self.values[key] = m.groups()

  def on_tick(self, elapsed):
    v = self.values
    if "iteration" not in v:
      text = f"setting up (compiling GPU kernels)...  {timedelta(seconds=int(elapsed))}"
    else:
      it, total = (int(x) for x in v["iteration"])
      frac = min(1.0, max((it + 1) / total, elapsed / self.budget_s))
      bar = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
      text = f"{bar} {frac * 100:5.1f}%  iter {it + 1}/{total}"
      if "reward" in v:
        text += f"  reward {float(v['reward'][0]):.2f}  ep len {float(v['length'][0]):.0f}/1000"
      if "terrain" in v:
        text += f"  terrain {float(v['terrain'][0]):.2f}"
      if "fps" in v:
        text += f"  {float(v['fps'][0]):,.0f} steps/s"
      text += f"  {timedelta(seconds=int(elapsed))} / {timedelta(seconds=int(self.budget_s))}"
    self.handle.update(ipd.HTML(f"<pre>{html.escape(text)}</pre>"))

progress = TrainProgress(TIME_BUDGET_HOURS * 3600)
sh(cmd, cwd=WORK, max_seconds=TIME_BUDGET_HOURS * 3600, log_file=WORK / "train.log",
   on_line=progress.on_line, on_tick=progress.on_tick)
""")

md("""
## 7. Training curves

Kaggle cannot show TensorBoard inline, so this plots the main curves from the event files.

* **Learning to walk on terrain:** `mean_episode_length` near 1000, `track_linear_velocity` high, `terrain_levels` rising.
  A dip when the speed range widens (iterations 1,500 and 4,000) is expected.
* **Learning the gait:** `contact_pattern`, `arm_leg_swing` and `leg_extension` rising, `gait_symmetry` shrinking towards 0.
""")
code("""
import matplotlib.pyplot as plt

def plot_curves():
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  run = sorted(p for p in LOG_ROOT.glob("*/") if list(p.glob("events.*")))[-1]
  acc = EventAccumulator(str(run)); acc.Reload()
  tags = acc.Tags()["scalars"]
  rows = [
    ["Train/mean_reward", "Train/mean_episode_length", "track_linear_velocity", "track_angular_velocity"],
    ["terrain_levels/mean", "terrain_levels/stairs_up", "terrain_levels/stairs_down", "terrain_levels/slope_up"],
    ["contact_pattern", "gait_symmetry", "arm_leg_swing", "leg_extension"],
  ]
  fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3 * len(rows)), squeeze=False)
  shown = set()
  for row, row_axes in zip(rows, axes):
    for name, ax in zip(row, row_axes):
      tag = next((t for t in tags if name in t), None)
      if tag is None:
        ax.set_axis_off()
        continue
      shown.add(tag)
      ev = acc.Scalars(tag)
      ax.plot([e.step for e in ev], [e.value for e in ev])
      ax.set_title(tag.split("/", 1)[-1], fontsize=9); ax.set_xlabel("iteration")
  fig.tight_layout()
  fig.savefig(run / "curves.png", dpi=120)
  plt.show()
  print("other tags:", [t for t in tags if t not in shown][:60])

try:
  plot_curves()
except Exception as e:  # Never fail the run here.
  print("Could not plot curves:", e)
""")

md("""
## 8. Results: does it work on every terrain?

Records the newest checkpoint on each terrain, at a fixed difficulty, on a straight-line command: 0.5 is the middle of
the curriculum and 1.0 its hardest level (15 cm stairs, 24° slopes). Each clip simulates `EVAL_ENVS` robots and the camera
follows the first one, not a hand-picked one, so the table describes all of them:

* **Fell:** share of the robots that fell at least once in the clip.
* **Speed:** mean forward speed against the command. A low speed with few falls means the robot is stuck or hesitating.
* **Result:** `PASS` when at most `MAX_FELL_PERCENT` fell and the speed reached `MIN_SPEED_RATIO` of the command.

Robots start on a random tile, so one near the edge of the 40 m area can finish its clip on flat ground; the
hard-terrain numbers are therefore slightly optimistic. This takes about 30–40 minutes. Everything is also in the zip in section 9.
""")
code("""
import json, math

EVAL_ENVS = 64
EVAL_SECONDS = 12
MAX_FELL_PERCENT = 10.0
MIN_SPEED_RATIO = 0.7

# (clip, terrain, difficulty, forward speed in m/s)
CLIPS = [("walk_flat", "flat", 0.0, 1.0), ("run_flat", "flat", 0.0, 2.5), ("run_rough", "random_rough", 0.5, 2.0)]
for terrain, speed in [("stairs_up", 0.8), ("stairs_down", 0.8), ("slope_up", 1.0), ("slope_down", 1.0),
                       ("random_rough", 1.0), ("waves", 1.0), ("obstacles", 0.8)]:
  CLIPS += [(f"walk_{terrain}_{level}", terrain, difficulty, speed) for level, difficulty in [("medium", 0.5), ("hard", 1.0)]]

def latest_checkpoint():
  runs = sorted(p for p in LOG_ROOT.glob("*/") if list(p.glob("model_*.pt")))
  ckpts = sorted(runs[-1].glob("model_*.pt"), key=iteration)
  print("checkpoint:", ckpts[-1])
  return ckpts[-1]

def evaluate_terrains():
  checkpoint = latest_checkpoint()
  results = []
  for clip, terrain, difficulty, speed in CLIPS:
    print("---", clip)
    stats_file = WORK / "videos" / "terrains" / f"{clip}.json"
    out = video(f"videos/terrains/{clip}.mp4", f"--checkpoint-file {checkpoint}", f"--terrain {terrain}",
                f"--difficulty {difficulty}", f"--forward-speed {speed}", f"--seconds {EVAL_SECONDS}",
                f"--num-envs {EVAL_ENVS}", f"--stats-file {stats_file}", "--width 854 --height 480", show=False)
    if out is None or not stats_file.exists():
      continue  # Reported as missing in the table.
    stats = json.loads(stats_file.read_text())
    passed = stats["fell_percent"] <= MAX_FELL_PERCENT and stats["mean_forward_speed"] >= MIN_SPEED_RATIO * speed
    results.append({"clip": clip, "path": str(out), "terrain": terrain, "difficulty": difficulty,
                    "command": speed, "passed": passed, **stats})
  (WORK / "videos" / "terrain_results.json").write_text(json.dumps(results, indent=1))
  return results

try:
  RESULTS = evaluate_terrains()
except Exception as e:
  RESULTS = []
  print("Could not evaluate the terrains:", e)
""")

md("### Overview: all terrains side by side")
code("""
RESULTS_JSON = WORK / "videos" / "terrain_results.json"
if RESULTS:
  sh(f"python {SCRIPTS / 'make_terrain_overview.py'} {RESULTS_JSON} {WORK / 'videos' / 'terrains_overview.mp4'}", check=False)
  overview = WORK / "videos" / "terrains_overview.mp4"
  if overview.exists():
    display(Video(str(overview), embed=True, width=960))
""")

md("### Results table")
code("""
from IPython.display import HTML

def results_table():
  found = {r["clip"]: r for r in RESULTS}
  rows = []
  for clip, terrain, difficulty, speed in CLIPS:
    r = found.get(clip)
    if r is None:
      rows.append(f"<tr><td>{clip}</td><td>{terrain}</td><td>{difficulty}</td><td colspan=4>no result</td></tr>")
      continue
    colour = "#1a7f37" if r["passed"] else "#cf222e"
    label = "PASS" if r["passed"] else "FAIL"
    camera = " (robot on camera fell)" if r["camera_robot_fell"] else ""
    rows.append(f"<tr><td>{clip}</td><td>{terrain}</td><td>{difficulty}</td><td>{r['fell_percent']:.0f} %{camera}</td>"
                f"<td>{r['mean_forward_speed']:.2f} / {speed} m/s</td>"
                f"<td style='color:{colour};font-weight:bold'>{label}</td></tr>")
  head = "<tr><th>Clip</th><th>Terrain</th><th>Difficulty</th><th>Fell</th><th>Speed / command</th><th>Result</th></tr>"
  display(HTML(f"<table>{head}{''.join(rows)}</table>"))
  passed = sum(r["passed"] for r in RESULTS)
  print(f"{passed} of {len(CLIPS)} clips passed ({len(CLIPS) - len(RESULTS)} without a result).")

results_table()
""")

md("### Every clip")
code("""
for r in RESULTS:
  print(f"{r['clip']}: {r['terrain']}, difficulty {r['difficulty']}, {r['command']} m/s, "
        f"{'PASS' if r['passed'] else 'FAIL'}, {r['fell_percent']:.0f} % of {r['num_envs']} robots fell")
  display(Video(r["path"], embed=True, width=640))
""")

md("""
## 9. Package the results

Writes `hu_d03_velocity_rough_<run>.zip` with the newest checkpoint, the ONNX policy, configs, curves and videos.
Download it from the notebook's *Output* tab. The full `logs/` folder is in the output too, for resuming.
""")
code("""
run = sorted(p for p in LOG_ROOT.glob("*/") if list(p.glob("model_*.pt")))[-1]
pack = WORK / "export" / run.name
shutil.rmtree(pack, ignore_errors=True)
pack.mkdir(parents=True)
newest = max(run.glob("model_*.pt"), key=iteration)
shutil.copy(newest, pack)
for f in [*run.glob("*.onnx"), *run.glob("params/*"), *run.glob("curves.png")]:
  dst = pack / f.relative_to(run)
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy(f, dst)
if (WORK / "videos").exists():
  shutil.copytree(WORK / "videos", pack / "videos", dirs_exist_ok=True)
archive = shutil.make_archive(str(WORK / f"{EXPERIMENT}_{run.name}"), "zip", pack)
print("wrote", archive)
sh(f"ls -lR {pack} | head -40")
""")

md("""
## 10. Download everything

Zips the whole `hu_d03_rl_rough` folder (all checkpoints, ONNX, TensorBoard logs, `train.log`, videos) into
`hu_d03_rl_rough_all.zip` and shows a link to download it straight to your computer. The link works in an
interactive session; for a saved version, download the zip from the *Output* tab instead.
""")
code("""
from IPython.display import FileLink

everything = Path("/kaggle/working/hu_d03_rl_rough_all")
Path(f"{everything}.zip").unlink(missing_ok=True)
shutil.make_archive(str(everything), "zip", WORK.parent, WORK.name)
size_mb = Path(f"{everything}.zip").stat().st_size / 1e6
print(f"wrote {everything}.zip ({size_mb:,.0f} MB)")
# FileLink paths are relative to the notebook's folder, /kaggle/working.
display(FileLink(os.path.relpath(f"{everything}.zip", os.getcwd())))
""")

md("""
## Tuning the gait

Change the weights in the `velocity_env_cfgs.py` cell above and start a new version. To keep the progress so far, add
the previous version as Input: a resume accepts new reward weights.

| What you see | Change in `hu_d03_rough_env_cfg` |
|---|---|
| Arms still stiff | Raise `arm_leg_swing` weight (0.75 → 1.5) |
| Crouched, bent knees | Raise `leg_extension` weight (0.5 → 1.0), or `target_height` (0.90 → 0.905) |
| Short, quick steps | Raise `air_time` weight (0.25 → 0.5) or `threshold_min` (0.1 → 0.2) |
| Limping | Stronger `gait_symmetry` weight (-2 → -5) |
| Falls a lot, terrain level stuck | Lower the gait weights, as they compete with balance; or postpone the speed stages |
| Never runs with a flight phase | Raise `contact_pattern` weight; lower `mechanical_power` |
""")

nb.save(OUT, {
  "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"},
})
