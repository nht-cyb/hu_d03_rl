"""Shared pieces for the notebook builders in this folder."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MJLAB_COMMIT = "8bb2bc9e187797689ec82c78c35019afd2fd0376"
DESCRIPTION_COMMIT = "0dd8454384c58991f3e70fd7613be29006d39f86"

# The hu_d03_mjlab package. tasks/__init__.py registers both task families, so
# the tracking config is needed even for a locomotion-only notebook.
PACKAGE_FILES = [
  "pyproject.toml",
  "src/hu_d03_mjlab/__init__.py",
  "src/hu_d03_mjlab/robot/__init__.py",
  "src/hu_d03_mjlab/robot/hu_d03_spec.py",
  "src/hu_d03_mjlab/robot/hu_d03_params.py",
  "src/hu_d03_mjlab/robot/hu_d03_constants.py",
  "src/hu_d03_mjlab/tasks/__init__.py",
  "src/hu_d03_mjlab/tasks/rl_cfgs.py",
  "src/hu_d03_mjlab/tasks/velocity_env_cfgs.py",
  "src/hu_d03_mjlab/tasks/tracking_env_cfgs.py",
]


class Notebook:
  def __init__(self):
    self.cells = []

  def md(self, text: str) -> None:
    self.cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip()})

  def code(self, text: str) -> None:
    self.cells.append({
      "cell_type": "code",
      "metadata": {},
      "execution_count": None,
      "outputs": [],
      "source": text.strip(),
    })

  def embed(self, files: list[str], dest: str) -> None:
    """One %%writefile cell per file, preceded by a cell creating the folders."""
    dirs = sorted({str(Path(f).parent) for f in files if Path(f).parent != Path(".")})
    self.code(
      f"for d in {dirs!r}:\n  (Path({dest!r}) / d).mkdir(parents=True, exist_ok=True)"
    )
    for rel in files:
      # %%writefile rejects an empty cell body, so empty files get a comment.
      text = (ROOT / rel).read_text().strip() or "# (empty)"
      self.code(f"%%writefile {dest}/{rel}\n{text}")

  def save(self, path: Path, metadata: dict) -> None:
    cells = [
      {**c, "source": [line + "\n" for line in c["source"].split("\n")[:-1]]
       + [c["source"].split("\n")[-1]]}
      for c in self.cells
    ]
    nb = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 0}
    path.write_text(json.dumps(nb, indent=1))
    print(f"wrote {path} ({len(cells)} cells)")
