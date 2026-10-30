"""Generate notebooks/train_hu_d03.ipynb.

The notebook is self-contained: the hu_d03_mjlab package and scripts are
embedded as %%writefile cells, so nothing has to be pushed to GitHub first.
Re-run this after editing anything under src/ or scripts/:

  python notebooks/build_notebook.py
"""

from nb_common import (
  DESCRIPTION_COMMIT,
  MJLAB_COMMIT,
  PACKAGE_FILES,
  ROOT,
  Notebook,
)

OUT = ROOT / "notebooks" / "train_hu_d03.ipynb"

EMBEDDED = PACKAGE_FILES + [
  "scripts/stand_test.py",
  "scripts/retarget_g1_to_hu_d03.py",
  "scripts/csv_to_npz_hu_d03.py",
  "scripts/record_video.py",
]

nb = Notebook()
md, code, cells = nb.md, nb.code, nb.cells


md("""
# LimX HU_D03: locomotion and mimic training with mjlab (Colab / Kaggle)

Trains two policies for the LimX Dynamics HU_D03 humanoid with
[mjlab](https://github.com/mujocolab/mjlab) on a free cloud GPU:

1. **Locomotion** (`Mjlab-Velocity-Flat-LimX-HU-D03`): follow forward/sideways/turn velocity commands.
2. **Mimic** (`Mjlab-Tracking-Flat-LimX-HU-D03`): track a reference motion from LAFAN1.

**Pick a GPU runtime first.** mjlab needs an NVIDIA GPU of the Turing generation or newer.

| | Colab | Kaggle |
|---|---|---|
| Runtime | *Runtime → Change runtime type → T4 GPU* | *Settings → Accelerator → GPU T4 x2* (**not P100**: too old for mjlab) |
| Internet | on by default | *Settings → Internet → On* (needs a phone-verified account) |
| Limits | about 12 h per session at most, often less; disconnects when idle | 12 h per session, about 30 GPU h per week |
| Where results are kept | your Google Drive (mounted below) | `/kaggle/working` (kept when you *Save Version*) |

Sessions end without warning, so checkpoints are saved every 50 (locomotion) or 100 (mimic) iterations
and every training cell can **resume** from the latest checkpoint.

**About the HU_D03 model.** The ankles and waist on the real robot are parallel linkages (two motors per joint pair).
This notebook replaces them with serial `ankle_pitch/roll` and `waist_roll/pitch` joints, as mjlab does for the
Unitree G1, so the policy outputs serial joint targets. On the real robot these must be converted to the A/B motor
commands. PD gains are not published by LimX and are derived from the rotor inertias (10 Hz, damping ratio 2).
""")

md("## 1. Settings")
code("""
import os, sys, subprocess
from pathlib import Path

ON_COLAB = "google.colab" in sys.modules
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if ON_COLAB:
  from google.colab import drive
  drive.mount("/content/drive")
  WORK = Path("/content/drive/MyDrive/hu_d03_rl")  # Logs and checkpoints survive disconnects.
elif ON_KAGGLE:
  WORK = Path("/kaggle/working/hu_d03_rl")
else:
  WORK = Path.home() / "hu_d03_rl"
CODE = Path("/tmp/hu_d03_code")  # Installed code (fast local disk, re-created each session).
WORK.mkdir(parents=True, exist_ok=True)
CODE.mkdir(parents=True, exist_ok=True)

# Training sizes. 4096 envs is what the PPO settings are tuned for; use 2048 if you run out of GPU memory.
NUM_ENVS = 4096
VELOCITY_ITERATIONS = 3000   # Raise and resume if the robot is not walking well yet.
TRACKING_ITERATIONS = 5000
USE_TWO_GPUS = ON_KAGGLE     # Kaggle "GPU T4 x2" gives two GPUs.

os.environ["WANDB_MODE"] = "disabled"   # Log to TensorBoard instead of Weights & Biases.
os.environ["MUJOCO_GL"] = "egl"
os.environ["MJLAB_WARP_QUIET"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"  # Stream training logs into the notebook.
os.environ["PYTHONFAULTHANDLER"] = "1"  # Print a Python traceback on a crash.
os.environ["HU_D03_DESCRIPTION_DIR"] = str(CODE / "humanoid-description" / "HU_D03_description")
print("work dir:", WORK)
""")

md("## 2. Check the GPU")
code("""
out = subprocess.run(
  ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
  capture_output=True, text=True,
)
if out.returncode != 0:
  raise SystemExit("No NVIDIA GPU found. Switch the runtime to a T4 GPU (see the table above).")
print(out.stdout)
name, cap, driver, _ = [s.strip() for s in out.stdout.splitlines()[0].split(",")]
if float(cap) < 7.5:
  raise SystemExit(f"{name} (compute {cap}) is older than Turing; mjlab needs a T4 or newer.")
DRIVER_MAJOR = int(driver.split(".")[0])
# mjlab's CUDA 13 build needs driver 580+; older drivers use its cu126 extra.
CUDA_EXTRA = "cu130" if DRIVER_MAJOR >= 580 else "cu126"
NUM_GPUS = len(out.stdout.splitlines())
print("mjlab extra:", CUDA_EXTRA, "| GPUs:", NUM_GPUS)
""")

md("## 3. Install mjlab and the LimX HU_D03 model (about 3–5 minutes)")
code(f"""
MJLAB_COMMIT = "{MJLAB_COMMIT}"
DESCRIPTION_COMMIT = "{DESCRIPTION_COMMIT}"

def sh(cmd):
  # Run a shell command, streaming its output into the notebook.
  print("$", cmd, flush=True)
  proc = subprocess.Popen(cmd, shell=True, text=True, bufsize=1,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  for line in proc.stdout:
    print(line, end="", flush=True)
  if proc.wait() != 0:
    raise SystemExit(f"command failed with exit code {{proc.returncode}}")

def fetch(repo, commit, dest):
  if not (dest / ".git").exists():
    sh(f"git init -q {{dest}} && git -C {{dest}} remote add origin {{repo}}")
  sh(f"git -C {{dest}} fetch -q --depth 1 origin {{commit}} && git -C {{dest}} checkout -q FETCH_HEAD")

fetch("https://github.com/mujocolab/mjlab.git", MJLAB_COMMIT, CODE / "mjlab")
fetch("https://github.com/limxdynamics/humanoid-description.git", DESCRIPTION_COMMIT, CODE / "humanoid-description")

sh("pip install -q uv")
sh(f"uv pip install --system -q -e '{{CODE / 'mjlab'}}[{{CUDA_EXTRA}}]'")
""")

md("""
### Write the HU_D03 package

These cells write the `hu_d03_mjlab` package (robot model and task configs) and the helper scripts.
They are generated from the `hu-d03-rl` folder; edit them there and re-run `notebooks/build_notebook.py`,
or edit them here directly.
""")
code("""
for d in ["src/hu_d03_mjlab/robot", "src/hu_d03_mjlab/tasks", "scripts"]:
  (CODE / "hu-d03-rl" / d).mkdir(parents=True, exist_ok=True)
""")
for rel in EMBEDDED:
  text = (ROOT / rel).read_text()
  code(f"%%writefile /tmp/hu_d03_code/hu-d03-rl/{rel}\n{text}")
code("""sh(f"uv pip install --system -q -e {CODE / 'hu-d03-rl'}")""")

md("""
## 4. Sanity checks

* The stand test holds the home pose with the PD gains in plain MuJoCo (CPU).
* `list-envs` should show the four `LimX-HU-D03` tasks.
* The zero-action video runs the full mjlab environment on the GPU. The robot should stand still.
""")
code("""
SCRIPTS = CODE / "hu-d03-rl" / "scripts"
sh(f"python {SCRIPTS / 'stand_test.py'} --seconds 10")
sh("list-envs | grep -i hu-d03")
""")
code("""
from IPython.display import Video

def video(task, out, *extra):
  sh(f"cd {WORK} && python {SCRIPTS / 'record_video.py'} {task} --output-file {out} " + " ".join(extra))
  return Video(str(WORK / out), embed=True, width=640)

video("Mjlab-Velocity-Flat-LimX-HU-D03", "videos/zero_action.mp4", "--agent zero", "--seconds 5")
""")

md("""
If rendering fails with an EGL error (seen on some Kaggle images), run this cell and then the video cell again:
""")
code("""
sh("apt-get -qq install -y libosmesa6 > /dev/null")
os.environ["MUJOCO_GL"] = "osmesa"
""")

md("""
## 5. Train the locomotion policy

Logs and checkpoints go to `WORK/logs/rsl_rl/hu_d03_velocity/<date_time>/`.
Each save also writes an ONNX copy of the policy for deployment.

**Resume after a disconnect:** re-run sections 1–3 (skip the checks), set `RESUME = True` and run this cell again.
It continues from the newest checkpoint of the newest run.
""")
code("""
RESUME = False

def train(task, iterations, *extra):
  gpus = '--gpu-ids "[0, 1]"' if USE_TWO_GPUS and NUM_GPUS >= 2 else ""
  resume = "--agent.resume True" if RESUME else ""
  sh(f"cd {WORK} && train {task} --env.scene.num-envs {NUM_ENVS} "
     f"--agent.max-iterations {iterations} --agent.logger tensorboard {gpus} {resume} " + " ".join(extra))

train("Mjlab-Velocity-Flat-LimX-HU-D03", VELOCITY_ITERATIONS)
""")

md("""
### Watch the training curves

On Colab, run this in a separate cell while training (or after). Look at `Episode_Reward/track_linear_velocity`
and the mean episode length: the robot is walking when the episode length reaches the 20 s limit.
""")
code("""
%load_ext tensorboard
%tensorboard --logdir {WORK}/logs/rsl_rl
""")

md("### Video of the trained locomotion policy")
code("""
def latest_checkpoint(experiment):
  runs = sorted((WORK / "logs" / "rsl_rl" / experiment).glob("*/"))
  ckpts = sorted(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  print("checkpoint:", ckpts[-1])
  return ckpts[-1]

video("Mjlab-Velocity-Flat-LimX-HU-D03", "videos/locomotion.mp4",
      f"--checkpoint-file {latest_checkpoint('hu_d03_velocity')}", "--seconds 10")
""")

md("""
## 6. Prepare mimic data

[GMR](https://github.com/YanjieZe/GMR) and similar retargeting tools do not support the HU_D03 yet, so this
notebook starts from the LAFAN1 motions that Unitree already retargeted to the G1
([lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset))
and copies them to the HU_D03 joint by joint (`scripts/retarget_g1_to_hu_d03.py`): joint signs are matched by
comparing joint axes, the elbows get a zero-pose offset, and the feet are kept above the ground.

Other clips: `dance1_subject1..3`, `dance2_subject1..5`, `run1_subject2`, `jumps1_subject1`, `fight1_subject2`, ...
LAFAN1 is licensed CC BY-NC-ND 4.0 (non-commercial).
""")
code("""
MOTION = "walk1_subject1"
LINE_RANGE = None  # e.g. (1, 1800) to use only the first 60 s of a long clip.

data = WORK / "motions"
data.mkdir(exist_ok=True)
url = f"https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/{MOTION}.csv"
if not (data / f"{MOTION}_g1.csv").exists():
  sh(f"wget -q -O {data / f'{MOTION}_g1.csv'} {url}")
sh(f"python {SCRIPTS / 'retarget_g1_to_hu_d03.py'} --input {data / f'{MOTION}_g1.csv'} "
   f"--output {data / f'{MOTION}_hu_d03.csv'} "
   f"--g1-xml {CODE / 'mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'}")
lr = f'--line-range "({LINE_RANGE[0]}, {LINE_RANGE[1]})"' if LINE_RANGE else ""
sh(f"python {SCRIPTS / 'csv_to_npz_hu_d03.py'} --input-file {data / f'{MOTION}_hu_d03.csv'} "
   f"--output-file {data / f'{MOTION}_hu_d03.npz'} {lr}")
MOTION_FILE = data / f"{MOTION}_hu_d03.npz"
""")
md("Replay the reference motion (no policy, terminations off) to check the retargeting looks right:")
code("""
video("Mjlab-Tracking-Flat-LimX-HU-D03", f"videos/{MOTION}_reference.mp4",
      f"--motion-file {MOTION_FILE}", "--agent zero", "--no-terminations True", "--seconds 10")
""")

md("""
## 7. Train the mimic policy

Same resume rule as above: set `RESUME = True` to continue the newest `hu_d03_tracking` run.
""")
code("""
RESUME = False
train("Mjlab-Tracking-Flat-LimX-HU-D03", TRACKING_ITERATIONS,
      f"--env.commands.motion.motion-file {MOTION_FILE}")
""")
code("""
video("Mjlab-Tracking-Flat-LimX-HU-D03", f"videos/{MOTION}_mimic.mp4",
      f"--checkpoint-file {latest_checkpoint('hu_d03_tracking')}",
      f"--motion-file {MOTION_FILE}", "--seconds 15")
""")

md("""
## 8. Download the results

Zips the newest run of each task (checkpoints, ONNX policies, configs, videos).
On Colab the files are also already in your Drive under `hu_d03_rl/`.
""")
code("""
import shutil
for experiment in ["hu_d03_velocity", "hu_d03_tracking"]:
  runs = sorted((WORK / "logs" / "rsl_rl" / experiment).glob("*/"))
  if runs:
    archive = shutil.make_archive(str(WORK / f"{experiment}_{runs[-1].name}"), "zip", runs[-1])
    print("wrote", archive)
    if ON_COLAB:
      from google.colab import files
      files.download(archive)
""")

nb.save(OUT, {
  "accelerator": "GPU",
  "colab": {"gpuType": "T4", "provenance": []},
  "kernelspec": {"display_name": "Python 3", "name": "python3"},
  "language_info": {"name": "python"},
})
