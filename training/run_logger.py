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
		except Exception:  
			pass
	return versions


class RunLogger:
	"""Creates the run folder and writes metrics to CSV files."""

	def __init__(
		self,
		experiment: str,
		seed: int,
		config: dict[str, Any],
		root: str | Path = "runs",
		device: str | None = None,
		checkpoint_path: str | Path = "checkpoints",
		flush_every: int = 1,
	) -> None:
		if flush_every < 1:
			raise ValueError("flush_every must be at least 1.")

		stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
		self.experiment = experiment
		self.seed = seed
		self.dir = Path(root) / experiment / f"{stamp}_seed{seed}"
		self.dir.mkdir(parents=True, exist_ok=False)

		self._checkpoint_path = Path(checkpoint_path)
		self._checkpoint_path.mkdir(parents=True, exist_ok=True)
		self._flush_every = flush_every
		self._metrics_buffer: list[list[Any]] = []

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
		return self._checkpoint_path

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
		"""Buffers one finished training episode; writes a batch every `flush_every` episodes."""
		self._metrics_buffer.append(
			[episode, env_steps, reward, length, _blank(epsilon), _blank(loss)]
		)
		if len(self._metrics_buffer) >= self._flush_every:
			self.flush()

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

	def flush(self) -> None:
		"""Writes any buffered training rows to metrics.csv in one go."""
		if self._metrics_file.closed or not self._metrics_buffer:
			return
		self._metrics.writerows(self._metrics_buffer)
		self._metrics_file.flush()
		self._metrics_buffer.clear()

	def finish(self, status: str = "completed") -> None:
		"""Writes remaining rows, closes the files and records duration and final status."""
		if self._metrics_file.closed:
			return
		self.flush()
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