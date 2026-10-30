"""Generate notebooks/kaggle_hu_d03_mimic.ipynb.

A Kaggle-only notebook that trains the HU_D03 mimic (motion tracking) policy
(Mjlab-Tracking-Flat-LimX-HU-D03) on a LAFAN1 clip. It is written to run unattended with
"Save Version -> Save & Run All (Commit)": training stops itself before
Kaggle's 12 h limit, every step after training is non-fatal, and a new version
resumes from the previous version's output. Re-run after editing src/ or
scripts/:

  python notebooks/build_kaggle_mimic.py
"""

from nb_common import (
  DESCRIPTION_COMMIT,
  MJLAB_COMMIT,
  PACKAGE_FILES,
  ROOT,
  Notebook,
)

OUT = ROOT / "notebooks" / "kaggle_hu_d03_mimic.ipynb"
CODE = "/kaggle/temp/hu_d03_code"

nb = Notebook()
md, code = nb.md, nb.code

md("""
# LimX HU_D03 mimic training on Kaggle (mjlab)

Trains `Mjlab-Tracking-Flat-LimX-HU-D03`: the HU_D03 humanoid learns to imitate a reference motion from
LAFAN1 (BeyondMimic-style motion tracking), with [mjlab](https://github.com/mujocolab/mjlab) and the
[LimX HU_D03 model](https://github.com/limxdynamics/humanoid-description).

### Before running

1. **Settings → Accelerator → GPU T4 x2.** Not P100: it is older than Turing and mjlab does not support it.
2. **Settings → Internet → On** (needs a phone-verified Kaggle account).
3. For BONES-SEED (`SOURCE = "seed"`, the default): request access at
   [huggingface.co/datasets/bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed), create a
   Hugging Face **read** token, and add it under **Add-ons → Secrets** as `HF_TOKEN` (tick it for this notebook).

### How to run

* **Try it interactively first:** run the cells down to *5. Prepare the motion* and make sure the
  reference replay looks like the clip (feet on the ground, arms and legs moving the right way).
  Interactive sessions stop when the browser is idle, so do not train there.
* **Train with Save Version → Save & Run All (Commit).** That runs the whole notebook in the background
  (up to 12 h) and keeps everything in `/kaggle/working`: motion file, checkpoints, ONNX policy, training curves,
  videos and a zip.
  Training stops itself after `TIME_BUDGET_HOURS` so the steps after it still run.

### Continue training in a new version

Open the notebook, *Add Input → Your Work → Notebooks*, and add **this notebook's previous version**.
The settings cell finds its newest checkpoint under `/kaggle/input` and resumes from it.
Keep `MOTION`, `MEDLEY` and `LINE_RANGE` the same as in that version: that is how the run is recognised.

### Several dances in one policy

List BONES-SEED filenames (or LAFAN1 clip names) in `MEDLEY`. Section 5 retargets each one, turns and shifts it
to start where the previous one ended, blends `BLEND_SECONDS` between them and trains one policy on the joined
clip. The policy follows whichever dance its reference shows, so it can perform any of them.
Training does not start from scratch either: with `WARM_START = True`, a new clip or medley starts from the newest
checkpoint of any earlier run added under Input (for example your single-dance run), as long as `TASK` is the same.
To add a dance later, append it to `MEDLEY`, add the previous version as Input, and run again: it warm-starts from
the previous medley.

### About the model

The real HU_D03 drives its ankles and waist through parallel linkages (two motors per joint pair). Here they are
serial `ankle_pitch/roll` and `waist_roll/pitch` joints, as mjlab does for the Unitree G1, so the policy's targets
must be converted to A/B motor commands on the real robot. LimX does not publish PD gains; they are derived from
the rotor inertias (10 Hz, damping ratio 2).

The reference motion is LAFAN1 as retargeted to the Unitree G1, copied to the HU_D03 joint by joint
(GMR does not support the HU_D03 yet), so foot and hand positions are approximate.
LAFAN1 is licensed CC BY-NC-ND 4.0 (non-commercial).
""")

md("## 1. Settings")
code(f"""
import json, os, shutil, signal, subprocess
from pathlib import Path

# Output folder in /kaggle/working, saved with each version. Use a new name per experiment to keep outputs apart
# (e.g. "hu_d03_rl" for the ballet run, "hu_d03_rl_medley" for the medley); resume and warm start find runs in any of them.
OUTPUT_NAME = "hu_d03_rl_medley"
WORK = Path("/kaggle/working") / OUTPUT_NAME
CODE = Path("{CODE}")    # Installed code, re-created each run.
WORK.mkdir(parents=True, exist_ok=True)
CODE.mkdir(parents=True, exist_ok=True)

NUM_ENVS = 4096            # Use 2048 if the GPU runs out of memory.
MAX_ITERATIONS = 10_000    # Upper bound; the time budget usually stops training first.
TIME_BUDGET_HOURS = 10.5   # Kaggle stops a version at 12 h; leave time for the video and zip.
USE_TWO_GPUS = False       # True splits envs over both T4s (faster, but less tested).
TRAIN = True               # False: skip training; only plot, record and package the newest run
                           # (this session's, or a previous version's added under Input).

# Code versions. These commits were tested with this notebook; "main" gets the latest.
MJLAB_REF = "{MJLAB_COMMIT}"
DESCRIPTION_REF = "{DESCRIPTION_COMMIT}"

# Reference motion. SOURCE picks the dataset:
#   "seed":   BONES-SEED mocap (dances incl. ballet). Needs access on Hugging Face and an HF_TOKEN secret (see top).
#             MOTION is a BONES-SEED filename; "" uses the longest match of SEED_SEARCH (section 1b lists them).
#   "lafan1": LAFAN1 retargeted to the G1, no sign-up. MOTION is a clip name: walk1_subject1, dance1_subject1..3,
#             dance2_subject1..5, run1_subject2, jumps1_subject1, fight1_subject2, ...
SOURCE = "seed"
MOTION = ""
SEED_SEARCH = "ballet"     # All words must appear in a motion's descriptions.
LINE_RANGE = None          # e.g. (1, 900) to use only the first 30 s (30 fps) of a long clip.

# Several dances in one policy: clip names of SOURCE (BONES-SEED below), joined in this order; MOTION is then
# ignored. Section 8 records one video per clip. [] trains on MOTION alone.
MEDLEY = [
  "walk_ff_loop_180_R_002__A046",                              # normal walking (16.9 s)
  "dance_hiphop_hip_hop_iv_R_002__A320",                       # fast arm and leg swings (16.3 s)
  "dance_latino_mambo_chase_180_turn_R_fast_002__A312",        # fast turns (11.1 s)
  "dance_vouge_the_duck_walk_variation_180_R_fast_002__A320",  # deep squat walk (9.3 s)
  "dance_latino_jive_sequence_R_101__A321",                    # kicks, quick footwork (25.4 s)
  "victory_dance_macarena_change_direction_R_loop_001__A321",  # jumps changing direction (28.5 s)
]
BLEND_SECONDS = 1.0        # Transition inserted between two clips of the medley.
WARM_START = True          # No earlier run of this clip/medley under Input: start from the newest checkpoint
                           # of any run there (same TASK) instead of from scratch.
WARM_START_FROM = ""       # "" picks automatically. To choose, give a model_*.pt (or a pattern) under /kaggle/input,
                           # e.g. "/kaggle/input/hu-d03-ballet/model_9999.pt" or "**/hu-d03-ballet/**/model_*.pt".
WARM_START_STD = 0.5       # Raise the warm-start policy's action noise to at least this (a fresh policy starts
                           # at 1.0; a trained one is ~0.2), so it explores the new dances. None keeps it.

TASK = "Mjlab-Tracking-Flat-LimX-HU-D03"
EXPERIMENT = "hu_d03_tracking"
LOG_ROOT = WORK / "logs" / "rsl_rl" / EXPERIMENT
EPISODE_STEPS = 500        # 10 s episodes at 50 Hz.
""")

md("""
### 1b. Pick the clip

For BONES-SEED this downloads the metadata (5 MB), lists the motions whose descriptions contain all words of
`SEED_SEARCH` (mirrored copies left out, longest first) and picks `MOTION`. To use another one, copy its
`filename` into `MOTION` in the settings and run both cells again.
""")
code(f"""
import re
if SOURCE == "seed":
  from huggingface_hub import hf_hub_download
  try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
  except Exception as e:
    print("No HF_TOKEN Kaggle secret (Add-ons → Secrets, then tick it for this notebook):", e)
  import pandas as pd
  meta = pd.read_parquet(hf_hub_download(
    "bones-studio/seed", "metadata/seed_metadata_v004.parquet", repo_type="dataset",
    local_dir=WORK / "motions" / "seed"))
  text_cols = [c for c in meta.columns if c.startswith("content_")]
  text = meta[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
  hits = meta[text.apply(lambda t: all(w in t for w in SEED_SEARCH.lower().split()))
              & ~meta["is_mirror"].fillna(False).astype(bool)]
  hits = hits.assign(seconds=hits["move_duration_frames"] / 120).sort_values("seconds", ascending=False)
  print(f"{{len(hits)}} motions match {{SEED_SEARCH!r}}")
  display(hits[["filename", "seconds", "category", "content_short_description",
                "content_natural_desc_4"]].head(40))
  if not MOTION and not MEDLEY:
    if hits.empty:
      raise SystemExit(f"Nothing matches {{SEED_SEARCH!r}}. Try other words (e.g. 'pirouette', 'plie', "
                       f"'arabesque', 'dance') or set MOTION to a filename from the BONES-SEED viewer.")
    MOTION = hits.iloc[0]["filename"]

MOTIONS = list(MEDLEY) or [MOTION]
if SOURCE == "seed":
  # Path of each clip's G1 CSV inside g1.tar.gz. The column name changes between metadata versions
  # (v004 has no move_g1_mujoco_path); without one, the archive is searched for <name>.csv anywhere.
  SEED_PATHS = {{}}
  for name in MOTIONS:
    rows = meta[meta["filename"] == name]
    if rows.empty:
      raise SystemExit(f"{{name!r}} is not a BONES-SEED filename (copy it from the table above).")
    row = rows.iloc[0]
    g1_cols = [c for c in meta.columns if "g1" in c.lower() and isinstance(row[c], str) and "/" in row[c]]
    SEED_PATHS[name] = row[g1_cols[0]] if g1_cols else None
    print(f"clip: {{name}} ({{row['move_duration_frames'] / 120:.1f}} s)\\n  {{row['content_natural_desc_4']}}")

# Clip name safe for file names and the run folder; a medley is named by a hash of its clip list.
if MEDLEY:
  import hashlib
  key = json.dumps([SOURCE, MOTIONS, BLEND_SECONDS])
  CLIP = f"medley{{len(MOTIONS)}}_{{hashlib.sha1(key.encode()).hexdigest()[:8]}}"
else:
  CLIP = re.sub(r"[^A-Za-z0-9_.-]+", "_", MOTION)
# Runs are named after the clip (logs/.../<date_time>_<RUN_NAME>) and only a run of the same clip is resumed.
RUN_NAME = CLIP + (f"_{{LINE_RANGE[0]}}-{{LINE_RANGE[1]}}" if LINE_RANGE else "")
print("run name:", RUN_NAME)

# Resume: newest checkpoint of this clip from an earlier version added under "Input".
def newest_input_run():
  ckpts = list(Path("/kaggle/input").glob(f"**/logs/rsl_rl/{{EXPERIMENT}}/*_{{RUN_NAME}}/model_*.pt"))
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
  print(f"no earlier {{RUN_NAME}} run under /kaggle/input")

# Warm start: otherwise begin from a checkpoint under Input: WARM_START_FROM, or else the newest model_*.pt there
# (an earlier version's logs, or a model you uploaded as a dataset, zipped or not). It must come from the same TASK
# (observation and action sizes must match). It is copied next to this run's logs and loaded by train.
def input_checkpoint():
  root = Path("/kaggle/input")
  if WARM_START_FROM:
    pattern = str(Path(WARM_START_FROM).relative_to(root)) if WARM_START_FROM.startswith("/") else WARM_START_FROM
    found = sorted(root.glob(pattern))
    if not found:
      raise SystemExit(f"WARM_START_FROM matches nothing under {{root}}. Checkpoints there: "
                       f"{{[str(p) for p in root.glob('**/model_*.pt')][:20]}}")
    return found[-1]
  ckpts = []
  for p in root.glob("**/model_*.pt"):
    if not re.fullmatch(r"model_\\d+\\.pt", p.name):
      continue
    # Skip runs of another TASK when their version's settings.json says so.
    settings = next((a / "settings.json" for a in p.parents if (a / "settings.json").exists()), None)
    if settings and json.loads(settings.read_text()).get("TASK") != TASK:
      continue
    ckpts.append(p)
  # Newest run (folders start with the date), then highest iteration.
  ckpts.sort(key=lambda p: (p.parent.name, int(p.stem.split("_")[1])))
  return ckpts[-1] if ckpts else None

INIT_RUN = INIT_CHECKPOINT = None
if RESUME_RUN is None and WARM_START:
  ckpt = input_checkpoint()
  if ckpt is not None:
    # A plain folder name: train's --agent.load-run is a regex and goes through the shell.
    INIT_RUN, INIT_CHECKPOINT = "warm_start_" + re.sub(r"[^A-Za-z0-9_-]+", "_", ckpt.parent.name), ckpt.name
    dest = LOG_ROOT / INIT_RUN
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt, dest / ckpt.name)
    print("warm start from", ckpt)
if RESUME_RUN is None and INIT_RUN is None:
  print("training from scratch")

# Saved so the cells after training can restore everything if the notebook restarts (see section 7).
(WORK / "settings.json").write_text(json.dumps(
  {{k: v for k, v in dict(globals()).items()
   if k.isupper() and isinstance(v, (str, int, float, bool, tuple, list, dict, type(None)))}}, indent=1))
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
md("""
Helper functions and environment variables. They are saved to `<OUTPUT_NAME>/nb_helpers.py` so the
restore cell in section 7 can load them again after a restart.
""")
code("""
%%writefile {WORK}/nb_helpers.py
import os, signal, subprocess, threading, time
from pathlib import Path
from IPython.display import Video, display

os.environ["WANDB_MODE"] = "disabled"
os.environ["MUJOCO_GL"] = "egl"
os.environ["MJLAB_WARP_QUIET"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"  # Stream training logs into the notebook.
os.environ["PYTHONFAULTHANDLER"] = "1"  # Print a Python traceback on a crash.
os.environ["HU_D03_DESCRIPTION_DIR"] = str(CODE / "humanoid-description" / "HU_D03_description")

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

def video(out, *extra, show=True):
  \"\"\"Record a video; returns None instead of failing the notebook. show=False: do not embed it.\"\"\"
  cmd = (f"python {SCRIPTS / 'record_video.py'} {TASK} --output-file {out} "
         f"--motion-file {MOTION_FILE} " + " ".join(extra))
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
""")
code("""exec((WORK / "nb_helpers.py").read_text())""")
code("""
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

Generated from the `hu-d03-rl` folder (`notebooks/build_kaggle_mimic.py`). You can also edit the cells here.
""")
nb.embed(PACKAGE_FILES + [
  "scripts/stand_test.py",
  "scripts/record_video.py",
  "scripts/retarget_g1_to_hu_d03.py",
  "scripts/csv_to_npz_hu_d03.py",
  "scripts/seed_g1_to_lafan_csv.py",
  "scripts/make_medley_hu_d03.py",
], f"{CODE}/hu-d03-rl")
code("""sh(f"uv pip install --system -q -e {CODE / 'hu-d03-rl'}")""")

md("""
## 4. Sanity checks

* The stand test holds the home pose in plain MuJoCo (CPU).
* `list-envs` must show the `LimX-HU-D03` tasks. If it does not, look for a
  `[WARN] Failed to load task package 'hu_d03'` traceback.
* Video problems never stop the run. If rendering fails with an EGL error, the video helper switches to
  OSMesa for that video and tries again.
""")
code("""
SCRIPTS = CODE / "hu-d03-rl" / "scripts"
sh(f"python {SCRIPTS / 'stand_test.py'} --seconds 10")
sh("list-envs | grep -i hu-d03")
""")

md("""
## 5. Prepare the motion

Gets the G1 clip (from `/kaggle/input` if a Kaggle dataset or earlier version there has it), converts a
BONES-SEED clip to the LAFAN1 layout (`seed_g1_to_lafan_csv.py`: cm → m, Euler → quaternion, 120 → 30 fps),
copies it to the HU_D03 joints (`retarget_g1_to_hu_d03.py`: joint signs from the joint axes, elbow zero-pose
offset, feet kept above the ground), then computes the body poses and velocities mjlab's tracking task needs
(`csv_to_npz_hu_d03.py`, on the GPU). The `.npz` is saved in the output next to the logs.

BONES-SEED ships all G1 clips in one 23.5 GB archive. It is streamed once and only the clips you picked are kept,
which takes roughly 5–30 minutes the first time; the extracted CSVs are saved in the output, so later versions
skip it. For a medley, the retargeted clips are then joined (`make_medley_hu_d03.py`, see *Several dances in one
policy* at the top) and `<clip>.segments.json` lists where each dance starts.
""")
code('''
import shlex
data = WORK / "motions"
data.mkdir(exist_ok=True)
safe = lambda name: re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

if SOURCE == "seed":
  missing = []
  for name in MOTIONS:
    seed_csv = data / f"{safe(name)}_seed.csv"
    local = sorted(Path("/kaggle/input").glob(f"**/{safe(name)}_seed.csv"))
    if not seed_csv.exists() and local:
      shutil.copy(local[0], seed_csv)
      print("using", local[0])
    if not seed_csv.exists():
      missing.append(name)
  if missing:
    # Stream the archive once and stop when every member matching <date>/<name>.csv has been found.
    members = [f"*{Path(SEED_PATHS[n]).parent.name}/{n}.csv" if SEED_PATHS[n] else f"*/{n}.csv" for n in missing]
    tmp = data / "seed_extract"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    url = "https://huggingface.co/datasets/bones-studio/seed/resolve/main/g1.tar.gz"
    print(f"streaming g1.tar.gz (23.5 GB) to extract {len(members)} clip(s) ...")
    sh(f'curl -sSfL -H "Authorization: Bearer $HF_TOKEN" {url} '
       f'| tar -xzf - -C {tmp} --wildcards --occurrence=1 {" ".join(map(shlex.quote, members))}', check=False)
    for name in missing:
      found = [p for p in tmp.rglob("*.csv") if p.name == f"{name}.csv"]
      if not found:
        raise SystemExit(f"{name}.csv not found in g1.tar.gz (path in metadata: {SEED_PATHS[name]}). "
                         "Check the HF_TOKEN secret and that your access request was accepted.")
      shutil.move(found[0], data / f"{safe(name)}_seed.csv")
    shutil.rmtree(tmp)

hu_csvs = []
for name in MOTIONS:
  g1_csv = data / f"{safe(name)}_g1.csv"
  if SOURCE == "seed":
    seed_csv = data / f"{safe(name)}_seed.csv"
    sh(f"python {SCRIPTS / 'seed_g1_to_lafan_csv.py'} --input {shlex.quote(str(seed_csv))} --output {g1_csv}")
  elif not g1_csv.exists():
    local = sorted(Path("/kaggle/input").glob(f"**/{name}.csv"))
    if local:
      shutil.copy(local[0], g1_csv)
      print("using", local[0])
    else:
      url = f"https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/{name}.csv"
      sh(f"wget -q -O {g1_csv}.part {url} && mv {g1_csv}.part {g1_csv}")
  hu_csvs.append(data / f"{safe(name)}_hu_d03.csv")
  sh(f"python {SCRIPTS / 'retarget_g1_to_hu_d03.py'} --input {g1_csv} --output {hu_csvs[-1]} "
     f"--g1-xml {CODE / 'mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'}")

if MEDLEY:
  hu_csv = data / f"{CLIP}_hu_d03.csv"
  sh(f"python {SCRIPTS / 'make_medley_hu_d03.py'} --inputs {' '.join(map(str, hu_csvs))} "
     f"--output {hu_csv} --blend-seconds {BLEND_SECONDS}")
else:
  hu_csv = hu_csvs[0]

MOTION_FILE = data / f"{RUN_NAME}_hu_d03.npz"
lr = f'--line-range "({LINE_RANGE[0]}, {LINE_RANGE[1]})"' if LINE_RANGE else ""
sh(f"python {SCRIPTS / 'csv_to_npz_hu_d03.py'} --input-file {hu_csv} --output-file {MOTION_FILE} {lr}")
print("motion file:", MOTION_FILE)
''')

md("""
Replay the reference (zero actions, terminations off) to check the retargeting. If the motion looks wrong here,
the policy cannot learn it: fix the clip before training.
""")
code("""
video(f"videos/{CLIP}_reference.mp4", "--agent zero", "--no-terminations True", "--seconds 10")
""")

md("""
## 6. Train

Logs, checkpoints (every 100 iterations) and ONNX exports go to `/kaggle/working/<OUTPUT_NAME>/logs/rsl_rl/hu_d03_tracking/<date_time>/`.
Training is stopped after `TIME_BUDGET_HOURS`; the newest checkpoint is kept.
Set `TRAIN = False` in the settings to skip this cell and only make the curves, video and zip of an existing run.

The cell shows one progress line, updated every 10 s. The percentage is of whichever limit comes first,
`MAX_ITERATIONS` or `TIME_BUDGET_HOURS`. The full rsl_rl output goes to `<OUTPUT_NAME>/train.log`
(`!tail -n 60 /kaggle/working/<OUTPUT_NAME>/train.log` shows the latest iteration in detail).
""")
code("""
cmd = (f"MUJOCO_GL=egl train {TASK} --env.scene.num-envs {NUM_ENVS} "
       f"--agent.max-iterations {MAX_ITERATIONS} --agent.logger tensorboard "
       f"--env.commands.motion.motion-file {MOTION_FILE} --agent.run-name {RUN_NAME}")
if USE_TWO_GPUS and NUM_GPUS >= 2:
  cmd += ' --gpu-ids "[0, 1]"'
if RESUME_RUN is not None:
  cmd += f" --agent.resume True --agent.load-run {RESUME_RUN.name}"
elif INIT_RUN is not None:
  cmd += f" --agent.resume True --agent.load-run {INIT_RUN} --agent.load-checkpoint {INIT_CHECKPOINT}"
  # Warm start: keep the weights but explore again and count iterations from 0 (edits the copy in LOG_ROOT).
  import torch
  ckpt_path = LOG_ROOT / INIT_RUN / INIT_CHECKPOINT
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  actor = ckpt["actor_state_dict"]
  if WARM_START_STD and "distribution.std_param" in actor:
    print(f"action std {actor['distribution.std_param'].mean():.2f} -> at least {WARM_START_STD}")
    actor["distribution.std_param"].clamp_(min=WARM_START_STD)
  for group in ckpt["optimizer_state_dict"]["param_groups"]:
    group["lr"] = 1.0e-3
  ckpt["iter"] = 0
  torch.save(ckpt, ckpt_path)
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
        text += f"  reward {float(v['reward'][0]):.2f}  ep len {float(v['length'][0]):.0f}/{EPISODE_STEPS}"
      if "fps" in v:
        text += f"  {float(v['fps'][0]):,.0f} steps/s"
      text += f"  {timedelta(seconds=int(elapsed))} / {timedelta(seconds=int(self.budget_s))}"
    self.handle.update(ipd.HTML(f"<pre>{html.escape(text)}</pre>"))

if TRAIN:
  progress = TrainProgress(TIME_BUDGET_HOURS * 3600)
  sh(cmd, cwd=WORK, max_seconds=TIME_BUDGET_HOURS * 3600, log_file=WORK / "train.log",
     on_line=progress.on_line, on_tick=progress.on_tick)
else:
  print("TRAIN = False: skipping training")
""")

md("""
### Restore after a restart

If the notebook restarted after training (idle timeout, crash, *Restart*), the variables and helpers from the
cells above are gone and the cells below would fail with `name '...' is not defined`. This cell loads them back
from `settings.json` and `nb_helpers.py`, without reinstalling or retraining. In an uninterrupted run it does nothing.
""")
code(f"""
if "video" not in globals():
  import json, os, shutil, subprocess
  from pathlib import Path
  # The output folder is the one whose settings.json was written last (OUTPUT_NAME in the settings).
  found = sorted(Path("/kaggle/working").glob("*/settings.json"), key=lambda p: p.stat().st_mtime)
  WORK = found[-1].parent if found else Path("/kaggle/working/hu_d03_rl")
  CODE = Path("{CODE}")
  if not (WORK / "settings.json").exists() or not (CODE / "mjlab").exists():
    raise SystemExit(
      "This is a new Kaggle session: nothing is installed. Set TRAIN = False and run all cells; "
      "sections 1-5 reinstall, then the cells below use the checkpoints in /kaggle/working "
      "or in a previous version added under Input.")
  globals().update(json.loads((WORK / "settings.json").read_text()))
  if LINE_RANGE:
    LINE_RANGE = tuple(LINE_RANGE)
  LOG_ROOT = WORK / "logs" / "rsl_rl" / EXPERIMENT
  SCRIPTS = CODE / "hu-d03-rl" / "scripts"
  MOTION_FILE = WORK / "motions" / f"{{RUN_NAME}}_hu_d03.npz"
  exec((WORK / "nb_helpers.py").read_text())
  print("restored", RUN_NAME, "| runs:", sorted(p.name for p in LOG_ROOT.glob(f"*_{{RUN_NAME}}/")))
""")

md("""
## 7. Training curves

Kaggle cannot show TensorBoard inline, so this plots the main curves from the event files.
The policy tracks the clip well when the episode length stays near the 10 s limit (no early terminations from
falling or drifting off the reference) and the `motion_body_pos` / `motion_global_root_pos` rewards are high.
""")
code("""
import matplotlib.pyplot as plt

def plot_curves():
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  run = sorted(p for p in LOG_ROOT.glob(f"*_{RUN_NAME}/") if list(p.glob("events.*")))[-1]
  acc = EventAccumulator(str(run)); acc.Reload()
  tags = acc.Tags()["scalars"]
  wanted = [t for t in tags if t in ("Train/mean_reward", "Train/mean_episode_length")]
  wanted += [t for t in tags if t.startswith("Episode_Reward/")
             and any(k in t for k in ("motion_body_pos", "motion_body_ori", "motion_global_root_pos"))]
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

md("""
## 8. Video of the trained policy

For a medley, one video per clip (`videos/01_<clip>_mimic.mp4`, ...): each starts from the reference pose at the
clip's first frame and runs to its end. The green ghost is the reference; if the robot falls, the episode restarts at
the beginning of the medley, so a sudden jump to another motion in a video means it fell.
""")
code("""
def latest_checkpoint():
  runs = sorted(p for p in LOG_ROOT.glob(f"*_{RUN_NAME}/") if list(p.glob("model_*.pt")))
  ckpts = sorted(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  print("checkpoint:", ckpts[-1])
  return ckpts[-1]

try:
  checkpoint = latest_checkpoint()
  if MEDLEY:
    # One video per clip of the medley, starting from the reference pose at the clip's first frame.
    # They are not embedded (the notebook would get too big): see the Output tab or the zip.
    from IPython.display import FileLink
    segments = json.loads((WORK / "motions" / f"{CLIP}_hu_d03.segments.json").read_text())
    for i, seg in enumerate(segments, 1):
      name = seg["clip"].removesuffix("_hu_d03")
      seconds = seg["end_s"] - seg["start_s"]
      print(f"\\n{i}/{len(segments)} {name}: {seg['start_s']:.1f}-{seg['end_s']:.1f} s")
      out = video(f"videos/{i:02d}_{name}_mimic.mp4", f"--checkpoint-file {checkpoint}",
                  f"--start-seconds {seg['start_s']}", f"--seconds {seconds:.1f}", show=False)
      if out:
        display(FileLink(os.path.relpath(out, os.getcwd())))
  else:
    video(f"videos/{CLIP}_mimic.mp4", f"--checkpoint-file {checkpoint}", "--seconds 15")
except Exception as e:
  print("Could not record the policy video:", e)
""")

md("""
## 9. Package the results

Writes `hu_d03_tracking_<run>.zip` with the newest checkpoint, the ONNX policy, the motion `.npz`, configs,
curves and videos.
Download it from the notebook's *Output* tab. The full `logs/` folder is in the output too, for resuming.
""")
code("""
run = sorted(p for p in LOG_ROOT.glob(f"*_{RUN_NAME}/") if list(p.glob("model_*.pt")))[-1]
pack = WORK / "export" / run.name
shutil.rmtree(pack, ignore_errors=True)
pack.mkdir(parents=True)
newest = max(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
shutil.copy(newest, pack)
for f in [*run.glob("*.onnx"), *run.glob("params/*"), *run.glob("curves.png")]:
  dst = pack / f.relative_to(run)
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy(f, dst)
shutil.copy(MOTION_FILE, pack)
for f in (WORK / "motions").glob(f"{CLIP}_hu_d03.segments.json"):   # Medley only: where each dance starts.
  shutil.copy(f, pack)
if (WORK / "videos").exists():
  shutil.copytree(WORK / "videos", pack / "videos", dirs_exist_ok=True)
archive = shutil.make_archive(str(WORK / f"{EXPERIMENT}_{run.name}"), "zip", pack)
print("wrote", archive)
sh(f"ls -lR {pack} | head -40")
""")

md("""
## 10. Download everything

Zips the whole `<OUTPUT_NAME>` folder (all checkpoints, ONNX, TensorBoard logs, `train.log`, videos) into
`<OUTPUT_NAME>_all.zip` and shows a link to download it straight to your computer. The link works in an
interactive session; for a saved version, download the zip from the *Output* tab instead.
""")
code("""
from IPython.display import FileLink

everything = WORK.parent / f"{WORK.name}_all"
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
