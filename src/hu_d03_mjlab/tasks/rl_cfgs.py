"""PPO runner configs for the HU_D03 tasks (G1 hyperparameters)."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def _runner_cfg(
  experiment_name: str, entropy_coef: float, save_interval: int
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=entropy_coef,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=save_interval,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def hu_d03_velocity_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _runner_cfg("hu_d03_velocity", entropy_coef=0.01, save_interval=50)


def hu_d03_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _runner_cfg("hu_d03_tracking", entropy_coef=0.005, save_interval=100)
