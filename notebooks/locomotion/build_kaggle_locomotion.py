"""Generate notebooks/kaggle_hu_d03_locomotion.ipynb.

A Kaggle-only notebook that trains the HU_D03 locomotion policy
(Mjlab-Velocity-Flat-LimX-HU-D03). It is written to run unattended with
"Save Version -> Save & Run All (Commit)": training stops itself before
Kaggle's 12 h limit, every step after training is non-fatal, and a new version
resumes from the previous version's output. Re-run after editing src/ or
scripts/:

  python notebooks/build_kaggle_locomotion.py
"""

from nb_common import (
  DESCRIPTION_COMMIT,
  MJLAB_COMMIT,
  PACKAGE_FILES,
  ROOT,
  Notebook,
)

OUT = ROOT / "notebooks" / "kaggle_hu_d03_locomotion.ipynb"
CODE = "/kaggle/temp/hu_d03_code"

nb = Notebook()
md, code = nb.md, nb.code

md("""
# LimX HU_D03 locomotion training on Kaggle (mjlab)

Trains `Mjlab-Velocity-Flat-LimX-HU-D03`: the HU_D03 humanoid learns to follow forward, sideways and turning
velocity commands on flat ground, with [mjlab](https://github.com/mujocolab/mjlab) and the
[LimX HU_D03 model](https://github.com/limxdynamics/humanoid-description).

### Before running

1. **Settings → Accelerator → GPU T4 x2.** Not P100: it is older than Turing and mjlab does not support it.
2. **Settings → Internet → On** (needs a phone-verified Kaggle account).

### How to run

* **Try it interactively first:** run the cells down to *4. Sanity checks* and make sure the stand video looks right.
  Interactive sessions stop when the browser is idle, so do not train there.
* **Train with Save Version → Save & Run All (Commit).** That runs the whole notebook in the background
  (up to 12 h) and keeps everything in `/kaggle/working`: checkpoints, ONNX policy, training curves, video and a zip.
  Training stops itself after `TIME_BUDGET_HOURS` so the steps after it still run.

### Continue training in a new version

Open the notebook, *Add Input → Your Work → Notebooks*, and add **this notebook's previous version**.
The settings cell finds its newest checkpoint under `/kaggle/input` and resumes from it.

### About the model

The real HU_D03 drives its ankles and waist through parallel linkages (two motors per joint pair). Here they are
serial `ankle_pitch/roll` and `waist_roll/pitch` joints, as mjlab does for the Unitree G1, so the policy's targets
must be converted to A/B motor commands on the real robot. LimX does not publish PD gains; they are derived from
the rotor inertias (10 Hz, damping ratio 2).
""")

md("## 1. Settings")
code(f"""
import os, shutil, signal, subprocess
from pathlib import Path

WORK = Path("/kaggle/working/hu_d03_rl")   # Saved with each version.
CODE = Path("{CODE}")    # Installed code, re-created each run.
WORK.mkdir(parents=True, exist_ok=True)
CODE.mkdir(parents=True, exist_ok=True)

NUM_ENVS = 4096            # Use 2048 if the GPU runs out of memory.
MAX_ITERATIONS = 10_000    # Upper bound; the time budget usually stops training first.
TIME_BUDGET_HOURS = 10.5   # Kaggle stops a version at 12 h; leave time for the video and zip.
USE_TWO_GPUS = False       # True splits envs over both T4s (faster, but less tested).

# Code versions. These commits were tested with this notebook; "main" gets the latest.
MJLAB_REF = "{MJLAB_COMMIT}"
DESCRIPTION_REF = "{DESCRIPTION_COMMIT}"

TASK = "Mjlab-Velocity-Flat-LimX-HU-D03"
EXPERIMENT = "hu_d03_velocity"
LOG_ROOT = WORK / "logs" / "rsl_rl" / EXPERIMENT

os.environ["WANDB_MODE"] = "disabled"
os.environ["MUJOCO_GL"] = "egl"
os.environ["MJLAB_WARP_QUIET"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"  # Stream training logs into the notebook.
os.environ["PYTHONFAULTHANDLER"] = "1"  # Print a Python traceback on a crash.
os.environ["HU_D03_DESCRIPTION_DIR"] = str(CODE / "humanoid-description" / "HU_D03_description")

# Resume: newest checkpoint of an earlier version added under "Input".
def newest_input_run():
  ckpts = list(Path("/kaggle/input").glob(f"**/logs/rsl_rl/{{EXPERIMENT}}/*/model_*.pt"))
  if not ckpts:
    return None
  ckpts.sort(key=lambda p: (p.parent.name, int(p.stem.split("_")[1])))
  return ckpts[-1].parent

RESUME_RUN = newest_input_run()
if RESUME_RUN is not None:
  dest = LOG_ROOT / RESUME_RUN.name
  if not dest.exists():
    shutil.copytree(RESUME_RUN, dest)
  print("will resume from", dest, "| newest checkpoint:",
        max(dest.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])).name)
else:
  print("no earlier run under /kaggle/input: training from scratch")
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

Generated from the `hu-d03-rl` folder (`notebooks/build_kaggle_locomotion.py`). You can also edit the cells here.
""")
nb.embed(PACKAGE_FILES + ["scripts/stand_test.py", "scripts/record_video.py"], f"{CODE}/hu-d03-rl")
code("""sh(f"uv pip install --system -q -e {CODE / 'hu-d03-rl'}")""")

md("""
## 4. Sanity checks

* The stand test holds the home pose in plain MuJoCo (CPU).
* `list-envs` must show the `LimX-HU-D03` tasks. If it does not, look for a
  `[WARN] Failed to load task package 'hu_d03'` traceback.
* The zero-action video runs the full mjlab environment on the GPU: the robot should stand still.
  If the error mentions EGL, the cell switches to OSMesa and tries again. Video problems never stop the run.
""")
code("""
SCRIPTS = CODE / "hu-d03-rl" / "scripts"
sh(f"python {SCRIPTS / 'stand_test.py'} --seconds 10")
sh("list-envs | grep -i hu-d03")
""")
code("""
from IPython.display import Video, display

def video(out, *extra):
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
  display(Video(str(WORK / out), embed=True, width=640))
  return WORK / out

video("videos/zero_action.mp4", "--agent zero", "--seconds 5")
""")

md("""
## 5. Train

Logs, checkpoints (every 50 iterations) and ONNX exports go to `/kaggle/working/hu_d03_rl/logs/rsl_rl/hu_d03_velocity/<date_time>/`.
Training is stopped after `TIME_BUDGET_HOURS`; the newest checkpoint is kept.

The cell shows one progress line, updated every 10 s. The percentage is of whichever limit comes first,
`MAX_ITERATIONS` or `TIME_BUDGET_HOURS`. The full rsl_rl output goes to `hu_d03_rl/train.log`
(`!tail -n 60 /kaggle/working/hu_d03_rl/train.log` shows the latest iteration in detail).
""")
code("""
cmd = (f"MUJOCO_GL=egl train {TASK} --env.scene.num-envs {NUM_ENVS} "
       f"--agent.max-iterations {MAX_ITERATIONS} --agent.logger tensorboard")
if USE_TWO_GPUS and NUM_GPUS >= 2:
  cmd += ' --gpu-ids "[0, 1]"'
if RESUME_RUN is not None:
  cmd += f" --agent.resume True --agent.load-run {RESUME_RUN.name}"
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
      if "fps" in v:
        text += f"  {float(v['fps'][0]):,.0f} steps/s"
      text += f"  {timedelta(seconds=int(elapsed))} / {timedelta(seconds=int(self.budget_s))}"
    self.handle.update(ipd.HTML(f"<pre>{html.escape(text)}</pre>"))

progress = TrainProgress(TIME_BUDGET_HOURS * 3600)
sh(cmd, cwd=WORK, max_seconds=TIME_BUDGET_HOURS * 3600, log_file=WORK / "train.log",
   on_line=progress.on_line, on_tick=progress.on_tick)
""")

md("""
## 6. Training curves

Kaggle cannot show TensorBoard inline, so this plots the main curves from the event files.
The robot walks well when the episode length stays near the 20 s limit and `track_linear_velocity` is high.
""")
code("""
import matplotlib.pyplot as plt

def plot_curves():
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  run = sorted(p for p in LOG_ROOT.glob("*/") if list(p.glob("events.*")))[-1]
  acc = EventAccumulator(str(run)); acc.Reload()
  tags = acc.Tags()["scalars"]
  wanted = [t for t in tags if t in ("Train/mean_reward", "Train/mean_episode_length")]
  wanted += [t for t in tags if "track_linear_velocity" in t or "track_angular_velocity" in t]
  fig, axes = plt.subplots(1, len(wanted), figsize=(4 * len(wanted), 3), squeeze=False)
  for ax, tag in zip(axes[0], wanted):
    ev = acc.Scalars(tag)
    ax.plot([e.step for e in ev], [e.value for e in ev])
    ax.set_title(tag.split("/")[-1], fontsize=9); ax.set_xlabel("iteration")
  fig.tight_layout()
  fig.savefig(run / "curves.png", dpi=120)
  plt.show()
  print("other tags:", [t for t in tags if t not in wanted][:40])

try:
  plot_curves()
except Exception as e:  # Never fail the run here.
  print("Could not plot curves:", e)
""")

md("## 7. Video of the trained policy")
code("""
def latest_checkpoint():
  runs = sorted(p for p in LOG_ROOT.glob("*/") if list(p.glob("model_*.pt")))
  ckpts = sorted(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  print("checkpoint:", ckpts[-1])
  return ckpts[-1]

try:
  video("videos/locomotion.mp4", f"--checkpoint-file {latest_checkpoint()}", "--seconds 15")
except Exception as e:
  print("Could not record the policy video:", e)
""")

md("""
## 8. Package the results

Writes `hu_d03_velocity_<run>.zip` with the newest checkpoint, the ONNX policy, configs, curves and videos.
Download it from the notebook's *Output* tab. The full `logs/` folder is in the output too, for resuming.
""")
code("""
run = sorted(p for p in LOG_ROOT.glob("*/") if list(p.glob("model_*.pt")))[-1]
pack = WORK / "export" / run.name
shutil.rmtree(pack, ignore_errors=True)
pack.mkdir(parents=True)
newest = max(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
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
## 9. Download everything

Zips the whole `hu_d03_rl` folder (all checkpoints, ONNX, TensorBoard logs, `train.log`, videos) into
`hu_d03_rl_all.zip` and shows a link to download it straight to your computer. The link works in an
interactive session; for a saved version, download the zip from the *Output* tab instead.
""")
code("""
from IPython.display import FileLink

everything = Path("/kaggle/working/hu_d03_rl_all")
Path(f"{everything}.zip").unlink(missing_ok=True)
shutil.make_archive(str(everything), "zip", WORK.parent, WORK.name)
size_mb = Path(f"{everything}.zip").stat().st_size / 1e6
print(f"wrote {everything}.zip ({size_mb:,.0f} MB)")
# FileLink paths are relative to the notebook's folder, /kaggle/working.
display(FileLink(os.path.relpath(f"{everything}.zip", os.getcwd())))
""")

nb.save(OUT, {
  "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"},
})
