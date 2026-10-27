"""
Aggregate experiment runs into a summary table and learning curve.

Usage:
	python -m evaluation.aggregate dqn_baseline
	python -m evaluation.aggregate dqn_baseline --window 50 --target 300
	python -m evaluation.aggregate dqn_baseline --include-unfinished

Reads:
	runs/<experiment>/*/{metrics.csv, eval.csv, run_info.json}

Writes:
	results/<experiment>/{summary.csv, summary.md, learning_curve.png}

Results are regenerated from run folders rather than manually copied values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
	"""Read CSV rows, returning an empty list when the file is missing."""
	if not path.exists():
		return []

	with path.open(newline="", encoding="utf-8") as file:
		return list(csv.DictReader(file))


def moving_average(
	values: np.ndarray,
	window: int,
) -> tuple[np.ndarray, int]:
	"""Return the trailing moving average and the actual window used."""
	if window < 1:
		raise ValueError("window must be at least 1.")

	if values.size == 0:
		return np.array([], dtype=float), 0

	used = min(window, len(values))
	kernel = np.ones(used, dtype=float) / used

	return np.convolve(values, kernel, mode="valid"), used


def load_run(
	run_dir: Path,
	window: int = 20,
	target: float | None = None,
) -> dict[str, Any] | None:
	"""Load and calculate metrics for one training run."""
	rows = _read_csv(run_dir / "metrics.csv")

	if not rows:
		return None

	steps = np.array([float(row["env_steps"]) for row in rows])
	rewards = np.array([float(row["reward"]) for row in rows])
	lengths = np.array([float(row["length"]) for row in rows])

	smooth, used = moving_average(rewards, window)
	smooth_steps = steps[used - 1 :]

	info_path = run_dir / "run_info.json"
	info = (
		json.loads(info_path.read_text(encoding="utf-8"))
		if info_path.exists()
		else {}
	)

	steps_to_target = None

	if target is not None:
		hit = np.flatnonzero(smooth >= target)

		if hit.size:
			steps_to_target = float(smooth_steps[hit[0]])

	eval_rows = _read_csv(run_dir / "eval.csv")
	eval_rewards = np.array(
		[
			float(row["reward"])
			for row in eval_rows
			if row.get("reward", "").strip()
		],
		dtype=float,
	)

	return {
		"name": run_dir.name,
		"seed": info.get("seed"),
		"status": info.get("status", "unknown"),
		"git_dirty": info.get("git_dirty"),
		"episodes": len(rows),
		"env_steps": float(steps[-1]),
		"final_reward": float(smooth[-1]),
		"best_episode": float(rewards.max()),
		"worst_episode": float(rewards.min()),
		"mean_length": float(lengths.mean()),
		"steps_to_target": steps_to_target,
		"train_seconds": info.get("duration_seconds"),
		"eval_mean": (
			float(eval_rewards.mean()) if eval_rewards.size else None
		),
		"eval_std": (
			float(eval_rewards.std()) if eval_rewards.size else None
		),
		"curve": (smooth_steps, smooth),
	}


def _mean_std(
	values: list[float | int | None],
) -> tuple[float, float] | None:
	"""Calculate mean and population standard deviation of valid values."""
	clean = [float(value) for value in values if value is not None]

	if not clean:
		return None

	return float(np.mean(clean)), float(np.std(clean))


def _fmt(
	mean_std: tuple[float, float] | None,
	digits: int = 1,
) -> str:
	if mean_std is None:
		return "n/a"

	mean, std = mean_std
	return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarize(
	runs: list[dict[str, Any]],
	target: float | None,
) -> list[tuple[str, str]]:
	"""Build aggregate metrics across training runs."""
	rows = [
		("runs (seeds)", str(len(runs))),
		(
			"final reward (last window)",
			_fmt(_mean_std([run["final_reward"] for run in runs])),
		),
		(
			"best episode reward",
			_fmt(_mean_std([run["best_episode"] for run in runs])),
		),
		(
			"worst episode reward",
			_fmt(_mean_std([run["worst_episode"] for run in runs])),
		),
		(
			"mean episode length",
			_fmt(_mean_std([run["mean_length"] for run in runs])),
		),
		(
			"environment steps trained",
			_fmt(_mean_std([run["env_steps"] for run in runs]), 0),
		),
		(
			"training time (seconds)",
			_fmt(_mean_std([run["train_seconds"] for run in runs]), 0),
		),
		(
			"evaluation reward (mean of eval episodes)",
			_fmt(_mean_std([run["eval_mean"] for run in runs])),
		),
	]

	if target is not None:
		reached = [
			run["steps_to_target"]
			for run in runs
			if run["steps_to_target"] is not None
		]

		rows.append(
			(f"runs reaching target {target:g}", f"{len(reached)}/{len(runs)}")
		)

		if reached:
			rows.append(
				(
					"environment steps to reach target",
					_fmt(_mean_std(reached), 0),
				)
			)

	return rows


def plot_learning_curve(
	runs: list[dict[str, Any]],
	experiment: str,
	out: Path,
) -> None:
	"""Plot individual run curves, their mean, and one-standard-deviation band."""
	if not runs:
		raise ValueError("Cannot plot a learning curve without runs.")

	import matplotlib

	matplotlib.use("Agg")

	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(figsize=(8, 5))

	grid_min = max(run["curve"][0][0] for run in runs)
	grid_max = min(run["curve"][0][-1] for run in runs)

	if grid_max > grid_min:
		grid = np.linspace(grid_min, grid_max, 300)

		curves = np.stack(
			[
				np.interp(grid, run["curve"][0], run["curve"][1])
				for run in runs
			]
		)

		mean = curves.mean(axis=0)
		std = curves.std(axis=0)

		for curve in curves:
			ax.plot(
				grid,
				curve,
				alpha=0.2,
				linewidth=0.8,
			)

		ax.plot(
			grid,
			mean,
			label=f"Mean of {len(runs)} runs",
		)

		ax.fill_between(
			grid,
			mean - std,
			mean + std,
			alpha=0.2,
			label="± 1 standard deviation",
		)
	else:
		# Curves with insufficient overlapping steps are shown individually.
		for run in runs:
			steps, rewards = run["curve"]
			ax.plot(steps, rewards, label=run["name"])

	ax.set_xlabel("Environment steps")
	ax.set_ylabel("Episode reward (moving average)")
	ax.set_title(experiment)
	ax.grid(alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(out, dpi=150)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Aggregate training runs and plot learning curves.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		 epilog="""\
			Examples:
			Basic aggregation: 
			uv run python -m evaluation.aggregate dqn_baseline

			Use the last 50 episodes for the moving average:
			uv run python -m evaluation.aggregate dqn_baseline --window 50

			Calculate environment steps to reach a reward of 300:
			uv run python -m evaluation.aggregate dqn_baseline --target 300

			Combine moving average and target:
			uv run python -m evaluation.aggregate dqn_baseline --window 50 --target 300

			Include interrupted, failed, and running experiments:
			uv run python -m evaluation.aggregate dqn_baseline --include-unfinished

			Specify custom directories:
			uv run python -m evaluation.aggregate dqn_baseline \\
				--runs-dir experiments --results-dir reports

			Output:
			results/<experiment>/summary.csv
			results/<experiment>/summary.md
			results/<experiment>/learning_curve.png
			""",
	)
	

	parser.add_argument("experiment", help="Experiment folder name")
	parser.add_argument("--runs-dir", default="runs")
	parser.add_argument("--results-dir", default="results")
	parser.add_argument(
		"--window",
		type=int,
		default=20,
		help="Episodes included in the trailing moving average",
	)
	parser.add_argument(
		"--target",
		type=float,
		default=None,
		help="Reward threshold for calculating steps-to-target",
	)
	parser.add_argument(
		"--include-unfinished",
		action="store_true",
		help="Include interrupted, failed, and running runs",
	)
	args = parser.parse_args()

	if args.window < 1:
		parser.error("--window must be at least 1.")

	experiment_dir = Path(args.runs_dir) / args.experiment

	if not experiment_dir.is_dir():
		raise SystemExit(f"Experiment directory not found: {experiment_dir}")

	run_dirs = sorted(
		path for path in experiment_dir.iterdir() if path.is_dir()
	)

	loaded_runs = [
		run
		for path in run_dirs
		if (run := load_run(path, args.window, args.target)) is not None
	]

	skipped = [
		run["name"]
		for run in loaded_runs
		if run["status"] != "completed"
	]

	if not args.include_unfinished:
		loaded_runs = [
			run for run in loaded_runs if run["status"] == "completed"
		]

	if not loaded_runs:
		raise SystemExit(
			f"No usable runs found in {experiment_dir}. "
			"Check metrics.csv files and run statuses."
		)

	out_dir = Path(args.results_dir) / args.experiment
	out_dir.mkdir(parents=True, exist_ok=True)

	table = summarize(loaded_runs, args.target)

	with (out_dir / "summary.csv").open(
		"w", newline="", encoding="utf-8"
	) as file:
		writer = csv.writer(file)
		writer.writerow(["metric", "mean ± std over runs"])
		writer.writerows(table)

	lines = [
		f"# {args.experiment}",
		"",
		"| Metric | Mean ± std over runs |",
		"|---|---|",
	]

	lines.extend(f"| {metric} | {value} |" for metric, value in table)

	lines.extend(
		[
			"",
			"## Individual runs",
			"",
			"| Run | Seed | Final reward | Eval mean | Status | Git dirty |",
			"|---|---:|---:|---:|---|---|",
		]
	)

	for run in loaded_runs:
		eval_mean = (
			"n/a"
			if run["eval_mean"] is None
			else f"{run['eval_mean']:.1f}"
		)

		lines.append(
			f"| {run['name']} | {run['seed']} | "
			f"{run['final_reward']:.1f} | {eval_mean} | "
			f"{run['status']} | {run['git_dirty']} |"
		)

	if skipped and not args.include_unfinished:
		lines.extend(
			[
				"",
				f"Excluded unfinished runs: {', '.join(skipped)}",
			]
		)

	if any(run["git_dirty"] for run in loaded_runs):
		lines.extend(
			[
				"",
				"Warning: some runs used uncommitted code changes.",
			]
		)

	(out_dir / "summary.md").write_text(
		"\n".join(lines) + "\n",
		encoding="utf-8",
	)

	plot_learning_curve(
		loaded_runs,
		args.experiment,
		out_dir / "learning_curve.png",
	)

	print("\n".join(lines))
	print(
		f"\nWrote {out_dir / 'summary.csv'}, "
		f"{out_dir / 'summary.md'}, "
		f"{out_dir / 'learning_curve.png'}"
	)


if __name__ == "__main__":
	main()