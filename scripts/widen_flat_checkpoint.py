"""Turn a Flat velocity checkpoint into a warm start for the Rough task.

The Rough observations add a terrain height scan, so the first layer of the actor
and critic is wider and a plain `--agent.resume` cannot load a Flat checkpoint.
This inserts the new inputs with zero weights: the widened policy walks exactly like
the Flat one until training teaches it to use the scan.

  python scripts/widen_flat_checkpoint.py results/policies/hu_d03_velocity_flat.pt \
      logs/rsl_rl/hu_d03_velocity_rough/0000_warm_start_flat/model_0.pt
"""

from pathlib import Path

import torch
import tyro

FLAT_ACTOR_OBS = 105
FLAT_CRITIC_OBS = 117
# Both groups list height_scan straight after the 105 shared terms; the critic's
# foot terms follow it. 17 x 11 rays: GridPatternCfg(size=(1.6, 1.0), resolution=0.1).
SCAN_START = 105
SCAN_RAYS = 187
# The scan is (base height above terrain) / 5 m, see make_velocity_env_cfg. The
# normalizer's real statistics take over during training; these keep the first
# iterations sensible (measured on the Rough terrains with the Flat policy).
SCAN_MEAN = 0.17
SCAN_STD = 0.03
# The Flat run's normalizer has seen 1e9 samples and would barely move on new terrain.
# At 4096 envs this adapts it over roughly the first 500 iterations.
NORMALIZER_COUNT = 50_000_000


def _insert(tensor: torch.Tensor, dim: int, value: float) -> torch.Tensor:
  head, tail = tensor.split([SCAN_START, tensor.shape[dim] - SCAN_START], dim)
  shape = list(tensor.shape)
  shape[dim] = SCAN_RAYS
  return torch.cat([head, tensor.new_full(shape, value), tail], dim)


def _widen(state: dict, flat_obs: int) -> None:
  width = state["mlp.0.weight"].shape[1]
  if width != flat_obs:
    raise SystemExit(f"Not a Flat checkpoint: {width} inputs instead of {flat_obs}.")
  state["mlp.0.weight"] = _insert(state["mlp.0.weight"], 1, 0.0)
  state["obs_normalizer._mean"] = _insert(state["obs_normalizer._mean"], 1, SCAN_MEAN)
  state["obs_normalizer._std"] = _insert(state["obs_normalizer._std"], 1, SCAN_STD)
  state["obs_normalizer._var"] = _insert(state["obs_normalizer._var"], 1, SCAN_STD**2)
  state["obs_normalizer.count"] = state["obs_normalizer.count"].clamp(
    max=NORMALIZER_COUNT
  )


def main(
  input_file: tyro.conf.Positional[str],
  output_file: tyro.conf.Positional[str],
  min_std: float = 0.3,
  learning_rate: float = 3.0e-4,
) -> None:
  """Write the widened checkpoint.

  Args:
    input_file: A Mjlab-Velocity-Flat-LimX-HU-D03 checkpoint.
    output_file: Where to write it, inside a run folder so `train` can resume it.
    min_std: Raise the action noise to at least this, to explore the new task.
    learning_rate: Starting learning rate; the adaptive schedule takes over.
  """
  ckpt = torch.load(input_file, map_location="cpu", weights_only=False)
  _widen(ckpt["actor_state_dict"], FLAT_ACTOR_OBS)
  _widen(ckpt["critic_state_dict"], FLAT_CRITIC_OBS)
  std = ckpt["actor_state_dict"]["distribution.std_param"]
  print(f"action std {std.min():.2f}..{std.max():.2f} -> at least {min_std}")
  std.clamp_(min=min_std)

  # Adam's moments have the old first-layer shape and belong to the old rewards.
  ckpt["optimizer_state_dict"]["state"] = {}
  for group in ckpt["optimizer_state_dict"]["param_groups"]:
    group["lr"] = learning_rate
  # A new training: from step 240,000 the command curriculum would start at its last stage.
  ckpt["iter"] = 0
  ckpt["infos"] = {"env_state": {"common_step_counter": 0}}

  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  torch.save(ckpt, output_file)
  print(f"wrote {output_file}")


if __name__ == "__main__":
  tyro.cli(main)
