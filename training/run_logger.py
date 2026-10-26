"""
Writes one self-describing folder per training run.

runs/<experiment>/<YYYY-MM-DD_HHMMSS>_seed<seed>/
	config.json     all hyperparameters used for the run
	run_info.json   seed, git commit, versions, device, start/end time, status
	metrics.csv     one row per training episode
	eval.csv        one row per evaluation episode
	checkpoint.pt   (path from `checkpoint_path`; you save it yourself)

Usage:

	config = {"lr": 1e-4, "batch_size": 64, "double_dqn": True}
	with RunLogger("dqn_baseline", seed=0, config=config) as run:
		for episode in ...:
			...
			run.log_episode(
				episode=episode, env_steps=total_steps, reward=ep_reward,
				length=ep_len, epsilon=epsilon, loss=mean_loss,
			)
		torch.save(agent.get_save_state(), run.checkpoint_path)
		for i, eval_seed in enumerate(eval_seeds):
			...
			run.log_eval_episode(
				env_steps=total_steps, eval_seed=eval_seed,
				reward=r, length=n,
			)
"""

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
		out = subprocess.run(
			["git", *args], capture_output=True, text=True, timeout=10
		)
	except (OSError, subprocess.SubprocessError):
		return None
	return out.stdout.strip() if out.returncode == 0 else None


def _library_versions() -> dict[str, str]:
	versions = {"python": platform.python_version()}
	for name in ("torch", "numpy", "gymnasium"):
		try:
			module = __import__(name)
			versions[name] = str(module.__version__)
		except Exception:  # not installed / no __version__
			pass
	return versions


class RunLogger:
	"""Creates the run folder and appends metrics to CSV files as they happen."""

	def __init__(
		self,
		experiment: str,
		seed: int,
		config: dict[str, Any],
		root: str | Path = "runs",
		device: str | None = "cpu",
	) -> None:
		"""
		Creates a new run folder and writes config.json and run_info.json.
		Arguments:
			experiment: name of the experiment (subfolder of `root`)
			seed: random seed used for this run
			config: hyperparameters and other config values to save
			root: root folder for all runs (default: "runs")
			device: device used for this run (default: "cpu")
		"""
		stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
		self.experiment = experiment
		self.seed = seed
		self.dir = Path(root) / experiment / f"{stamp}_seed{seed}"
		self.dir.mkdir(parents=True, exist_ok=False)

		# Ignore our own outputs (runs/, results/) when checking for uncommitted
		# code, otherwise every earlier run would make the next one "dirty".
		status = _git(
			"status", "--porcelain", "--", ".",
			f":(exclude){root}", ":(exclude)results",
		)

		self._start = time.monotonic()
		self._info: dict[str, Any] = {
			"experiment": experiment,
			"seed": seed,
			"started_at": datetime.now().isoformat(timespec="seconds"),
			"status": "running",
			"git_commit": _git("rev-parse", "HEAD"),
			"git_dirty": None if status is None else bool(status),
			"device": device,
			"platform": platform.platform(),
			"versions": _library_versions(),
			"command": " ".join(sys.argv),
		}

		self._write_json("config.json", config)
		self._write_json("run_info.json", self._info)

		self._metrics_file, self._metrics = self._open_csv(
			"metrics.csv", METRIC_FIELDS
		)
		self._eval_file, self._eval = self._open_csv("eval.csv", EVAL_FIELDS)

	@property
	def checkpoint_path(self) -> Path:
		return self.dir / "checkpoint.pt"

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
		"""Logs one finished training episode."""
		self._metrics.writerow(
			[episode, env_steps, reward, length, _blank(epsilon), _blank(loss)]
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
		"""Logs one evaluation episode (run with a fixed, separate seed)."""
		self._eval.writerow([env_steps, eval_seed, reward, length])
		self._eval_file.flush()

	def finish(self, status: str = "completed") -> None:
		"""Closes the files and records duration and final status."""
		if self._metrics_file.closed:
			return
		self._info["status"] = status
		self._info["finished_at"] = datetime.now().isoformat(timespec="seconds")
		self._info["duration_seconds"] = round(time.monotonic() - self._start, 1)
		self._write_json("run_info.json", self._info)
		self._metrics_file.close()
		self._eval_file.close()

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

	def _write_json(self, name: str, data: dict[str, Any]) -> None:
		(self.dir / name).write_text(
			json.dumps(data, indent=2, default=str) + "\n"
		)

	def _open_csv(self, name: str, fields: tuple[str, ...]):
		file = open(self.dir / name, "w", newline="")
		writer = csv.writer(file)
		writer.writerow(fields)
		file.flush()
		return file, writer


def _blank(value: float | None) -> float | str:
	return "" if value is None else value



# use example:
# import random
# import gymnasium as gym
# from training.run_logger import RunLogger

# SEED = 0
# config = {"lr": 1e-4, "gamma": 0.99, "batch_size": 64,
#           "eps_decay_steps": 50_000, "total_steps": 100_000}

# env = gym.make("CartPole-v1")      # swap for CarRacing later
# agent = DQNetworkTrainAgent(...)   # your network, optimizer, double_dqn=True, etc.

# with RunLogger("dqn_baseline", seed=SEED, config=config) as run:
# 	state, _ = env.reset(seed=SEED)
# 	total_steps, episode, ep_reward, ep_len, losses = 0, 0, 0.0, 0, []

# 	while total_steps < config["total_steps"]:
# 		epsilon = max(0.05, 1 - total_steps / config["eps_decay_steps"])
# 		if random.random() < epsilon:
# 			action = env.action_space.sample()
# 		else:
# 			action, _ = agent.get_action(state)

# 		next_state, reward, terminated, truncated, _ = env.step(action)
# 		loss = agent.update_weights(state, action, reward, next_state, terminated)
# 		if agent.can_learn:
# 			losses.append(loss)

# 		state = next_state
# 		total_steps += 1
# 		ep_reward += reward
# 		ep_len += 1

# 		if terminated or truncated:
# 			run.log_episode(
# 				episode=episode, env_steps=total_steps, reward=ep_reward,
# 				length=ep_len, epsilon=epsilon,
# 				loss=sum(losses) / len(losses) if losses else None,
# 			)
# 			state, _ = env.reset()
# 			episode, ep_reward, ep_len, losses = episode + 1, 0.0, 0, []

# 	torch.save(agent.get_save_state(), run.checkpoint_path)