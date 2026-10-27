from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self


METRIC_FIELDS = ("episode", "env_steps", "reward", "length", "epsilon", "loss")
EVAL_FIELDS = ("env_steps", "eval_seed", "reward", "length")


def _git(*args: str) -> str | None:
	try:
		result = subprocess.run(
			["git", *args],
			capture_output=True,
			text=True,
			timeout=10,
			check=False,
		)
	except (OSError, subprocess.SubprocessError):
		return None

	return result.stdout.strip() if result.returncode == 0 else None


def _library_versions() -> dict[str, str]:
	versions = {"python": platform.python_version()}

	for name in ("torch", "numpy", "gymnasium"):
		try:
			module = __import__(name)
			versions[name] = str(module.__version__)
		except Exception:
			pass

	return versions


def _blank(value: float | int | None) -> float | int | str:
	return "" if value is None else value


class RunLogger:
	"""Create a self-describing folder for one training run."""

	def __init__(
		self,
		experiment: str,
		seed: int = 0,
		config: dict[str, Any] | None = None,
		*,
		root: str | Path = "runs",
		device: str | None = "cpu",
		metadata: dict[str, Any] | None = None,
		checkpoint_name: str = "checkpoint.pt",
		**hyperparameters: Any,
	) -> None:
		"""
		Create a run directory and initialize its log files.

		Args:
			experiment: Experiment name.
			seed: Random seed, defaulting to 0.
			config: Optional dictionary of hyperparameters.
			root: Root directory for all experiments.
			device: Device used for training.
			metadata: Additional run metadata saved to run_info.json.
			checkpoint_name: Checkpoint filename.
			**hyperparameters: Additional hyperparameters saved to config.json.

		Explicit hyperparameters override matching keys in config.
		"""
		if not experiment.strip():
			raise ValueError("experiment cannot be empty.")

		if Path(checkpoint_name).name != checkpoint_name:
			raise ValueError("checkpoint_name must be a filename, not a path.")

		self.experiment = experiment
		self.seed = int(seed)
		self.checkpoint_name = checkpoint_name

		# Merge configuration without modifying the caller's dictionary.
		self.config = dict(config or {})
		self.config.update(hyperparameters)

		stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
		self.dir = Path(root) / experiment / f"{stamp}_seed{self.seed}"
		self.dir.mkdir(parents=True, exist_ok=False)

		# Ignore generated output directories when checking repository status.
		root_path = Path(root)
		excluded_paths = [f":(exclude){root_path.as_posix()}"]

		if root_path.as_posix() != "results":
			excluded_paths.append(":(exclude)results")

		status = _git("status", "--porcelain", "--", ".", *excluded_paths)

		self._start = time.monotonic()
		self._finished = False

		self._info: dict[str, Any] = {
			"experiment": experiment,
			"seed": self.seed,
			"started_at": datetime.now().isoformat(timespec="seconds"),
			"status": "running",
			"git_commit": _git("rev-parse", "HEAD"),
			"git_dirty": None if status is None else bool(status),
			"device": device,
			"platform": platform.platform(),
			"versions": _library_versions(),
			"command": " ".join(sys.argv),
			**(metadata or {}),
		}

		# Core run fields should not be accidentally replaced by metadata.
		self._info.update(
			experiment=experiment,
			seed=self.seed,
			status="running",
		)

		self._write_json("config.json", self.config)
		self._write_json("run_info.json", self._info)

		self._metrics_file, self._metrics = self._open_csv(
			"metrics.csv", METRIC_FIELDS
		)
		self._eval_file, self._eval = self._open_csv(
			"eval.csv", EVAL_FIELDS
		)

	@property
	def checkpoint_path(self) -> Path:
		"""Path where the caller can save its model checkpoint."""
		return self.dir / self.checkpoint_name

	@property
	def run_info(self) -> dict[str, Any]:
		"""Read-only-by-convention access to the current run metadata."""
		return self._info.copy()

	def log_episode(
		self,
		*,
		episode: int,
		env_steps: int,
		reward: float,
		length: int,
		epsilon: float | None = None,
		loss: float | None = None,
	) -> None:
		"""Append one completed training episode."""
		self._ensure_open()

		self._metrics.writerow(
			[
				episode,
				env_steps,
				reward,
				length,
				_blank(epsilon),
				_blank(loss),
			]
		)
		self._metrics_file.flush()

	def log_eval_episode(
		self,
		*,
		env_steps: int,
		eval_seed: int,
		reward: float,
		length: int,
	) -> None:
		"""Append one evaluation episode."""
		self._ensure_open()

		self._eval.writerow([env_steps, eval_seed, reward, length])
		self._eval_file.flush()

	def finish(self, status: str = "completed") -> None:
		"""Close the logs and persist the final run status."""
		if self._finished:
			return

		self._info["status"] = status
		self._info["finished_at"] = datetime.now().isoformat(
			timespec="seconds"
		)
		self._info["duration_seconds"] = round(
			time.monotonic() - self._start, 2
		)

		self._write_json("run_info.json", self._info)

		self._metrics_file.close()
		self._eval_file.close()
		self._finished = True

	def __enter__(self) -> Self:
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc: BaseException | None,
		tb: TracebackType | None,
	) -> None:
		if exc_type is None:
			self.finish("completed")
		elif issubclass(exc_type, KeyboardInterrupt):
			self.finish("interrupted")
		else:
			self.finish("failed")

	def _ensure_open(self) -> None:
		if self._finished:
			raise RuntimeError("Cannot log to a finished run.")

	def _write_json(self, name: str, data: dict[str, Any]) -> None:
		path = self.dir / name
		path.write_text(
			json.dumps(data, indent=2, default=str) + "\n",
			encoding="utf-8",
		)

	def _open_csv(
		self,
		name: str,
		fields: tuple[str, ...],
	) -> tuple[Any, Any]:
		file = open(self.dir / name, "w", newline="", encoding="utf-8")
		writer = csv.writer(file)
		writer.writerow(fields)
		file.flush()
		return file, writer

	


# import random

# import gymnasium as gym
# import torch

# from training.run_logger import RunLogger


# SEED = 0

# # These values are saved automatically in config.json.
# config = {
#     "learning_rate": 1e-4,
#     "gamma": 0.99,
#     "batch_size": 64,
# }

# with RunLogger(
#     "dqn_baseline",
#     seed=SEED,
#     config=config,
#     device="cpu",
#     root="runs",
#     total_steps=100_000,
#     eps_decay_steps=50_000,
#     double_dqn=True,
# ) as run:
#     env = gym.make("CartPole-v1")
#     env.action_space.seed(SEED)

#     # Replace with your own agent.
#     agent = DQNetworkTrainAgent(...)

#     state, _ = env.reset(seed=SEED)

#     total_steps = 0
#     episode = 0
#     ep_reward = 0.0
#     ep_len = 0
#     losses = []

#     while total_steps < run.config["total_steps"]:
#         epsilon = max(
#             0.05,
#             1.0 - total_steps / run.config["eps_decay_steps"],
#         )

#         if random.random() < epsilon:
#             action = env.action_space.sample()
#         else:
#             action, _ = agent.get_action(state)

#         next_state, reward, terminated, truncated, _ = env.step(action)

#         loss = agent.update_weights(
#             state, action, reward, next_state, terminated
#         )

#         if agent.can_learn and loss is not None:
#             losses.append(float(loss))

#         state = next_state
#         total_steps += 1
#         ep_reward += reward
#         ep_len += 1

#         if terminated or truncated:
#             run.log_episode(
#                 episode=episode,
#                 env_steps=total_steps,
#                 reward=ep_reward,
#                 length=ep_len,
#                 epsilon=epsilon,
#                 loss=sum(losses) / len(losses) if losses else None,
#             )

#             state, _ = env.reset()
#             episode += 1
#             ep_reward = 0.0
#             ep_len = 0
#             losses = []

#     torch.save(agent.get_save_state(), run.checkpoint_path)
#     env.close()