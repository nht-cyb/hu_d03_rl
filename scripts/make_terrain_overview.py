"""Tile the terrain clips into one labelled video, green for PASS and red for FAIL.

  python scripts/make_terrain_overview.py videos/terrain_results.json videos/terrains_overview.mp4
"""

import json
import math
from pathlib import Path

import mediapy
import numpy as np
import tyro
from PIL import Image, ImageDraw

import mjlab  # noqa: F401  (Points mediapy at its ffmpeg.)

CELL_WIDTH, CELL_HEIGHT = 320, 180
COLUMNS = 4
FRAME_STRIDE = 2  # Half the frames: keeps the file small.


def _cell(frame: np.ndarray, label: str, passed: bool) -> np.ndarray:
  image = Image.fromarray(frame).resize((CELL_WIDTH, CELL_HEIGHT))
  draw = ImageDraw.Draw(image)
  draw.rectangle((0, 0, CELL_WIDTH, 14), fill=(255, 255, 255))
  draw.text((3, 1), label, fill=(26, 127, 55) if passed else (207, 34, 46))
  return np.asarray(image)


def main(
  results_file: tyro.conf.Positional[str], output_file: tyro.conf.Positional[str]
) -> None:
  results = json.loads(Path(results_file).read_text())
  readers = [mediapy.VideoReader(r["path"]) for r in results]
  for reader in readers:
    reader.__enter__()
  fps = readers[0].metadata.fps / FRAME_STRIDE
  rows = math.ceil(len(results) / COLUMNS)
  blank = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
  labels = [
    f"{r['clip']}  fell {r['fell_percent']:.0f}%  {r['mean_forward_speed']:.1f}/{r['command']} m/s"
    for r in results
  ]
  shape = (rows * CELL_HEIGHT, COLUMNS * CELL_WIDTH)
  with mediapy.VideoWriter(output_file, shape=shape, fps=fps) as writer:
    for i, frames in enumerate(zip(*readers)):
      if i % FRAME_STRIDE:
        continue
      cells = [
        _cell(f, label, r["passed"]) for f, label, r in zip(frames, labels, results)
      ]
      cells += [blank] * (rows * COLUMNS - len(cells))
      writer.add_image(
        np.vstack([
          np.hstack(cells[j : j + COLUMNS]) for j in range(0, len(cells), COLUMNS)
        ])
      )
  for reader in readers:
    reader.__exit__(None, None, None)
  print(f"wrote {output_file}")


if __name__ == "__main__":
  tyro.cli(main)
