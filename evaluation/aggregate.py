"""
Turns the runs of one experiment into a results table and a learning curve.

	python -m evaluation.aggregate dqn_baseline
	python -m evaluation.aggregate dqn_baseline --window 50 --target 300

Reads  runs/<experiment>/*/{metrics.csv, eval.csv, run_info.json}
Writes results/<experiment>/{summary.csv, summary.md, learning_curve.png}

Everything is computed from the run folders, so results can be regenerated at
any time. Never copy numbers into the report by hand.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
	if not path.exists():
		return []
	with open(path, newline="") as f:
		return list(csv.DictReader(f))


def moving_average(x: np.ndarray, window: int) -> tuple[np.ndarray, int]:
	"""Trailing moving average. Returns (smoothed, window actually used)."""
	window = max(1, min(window, len(x)))
	kernel = np.ones(window) / window
	return np.convolve(x, kernel, mode="valid"), window


def load_run(run_dir: Path, window: int, target: float | None) -> dict | None:
	rows = _read_csv(run_dir / "metrics.csv")
	if not rows:
		return None

	steps = np.array([float(r["env_steps"]) for r in rows])
	rewards = np.array([float(r["reward"]) for r in rows])
	lengths = np.array([float(r["length"]) for r in rows])
	smooth, used = moving_average(rewards, window)
	smooth_steps = steps[used - 1 :]

	info_path = run_dir / "run_info.json"
	info = json.loads(info_path.read_text()) if info_path.exists() else {}

	steps_to_target = None
	if target is not None:
		hit = np.nonzero(smooth >= target)[0]
		if len(hit):
			steps_to_target = float(smooth_steps[hit[0]])

	eval_rows = _read_csv(run_dir / "eval.csv")
	eval_rewards = np.array([float(r["reward"]) for r in eval_rows])

	return {
		"name": run_dir.name,
		"seed": info.get("seed"),
		"status": info.get("status", "unknown"),
		"git_dirty": info.get("git_dirty"),
		"episodes": len(rows),
		"env_steps": float(steps[-1]),
		"final_reward": float(smooth[-1]),  # mean of the last `window` episodes
		"best_episode": float(rewards.max()),
		"worst_episode": float(rewards.min()),
		"mean_length": float(lengths.mean()),
		"steps_to_target": steps_to_target,
		"train_seconds": info.get("duration_seconds"),
		"eval_mean": float(eval_rewards.mean()) if len(eval_rewards) else None,
		"eval_std": float(eval_rewards.std()) if len(eval_rewards) else None,
		"curve": (smooth_steps, smooth),
	}


def _mean_std(values: list[float | None]) -> tuple[float, float] | None:
	clean = [v for v in values if v is not None]
	if not clean:
		return None
	return float(np.mean(clean)), float(np.std(clean))


def _fmt(ms: tuple[float, float] | None, digits: int = 1) -> str:
	return "n/a" if ms is None else f"{ms[0]:.{digits}f} ± {ms[1]:.{digits}f}"


def summarize(runs: list[dict], target: float | None) -> list[tuple[str, str]]:
	"""Rows of (metric, 'mean ± std over runs')."""
	rows = [
		("runs (seeds)", str(len(runs))),
		("final reward (last window)", _fmt(_mean_std([r["final_reward"] for r in runs]))),
		("best episode reward", _fmt(_mean_std([r["best_episode"] for r in runs]))),
		("worst episode reward", _fmt(_mean_std([r["worst_episode"] for r in runs]))),
		("mean episode length", _fmt(_mean_std([r["mean_length"] for r in runs]))),
		("env steps trained", _fmt(_mean_std([r["env_steps"] for r in runs]), 0)),
		("train time (s)", _fmt(_mean_std([r["train_seconds"] for r in runs]), 0)),
		("eval reward (mean of eval episodes)", _fmt(_mean_std([r["eval_mean"] for r in runs]))),
	]
	if target is not None:
		reached = [r["steps_to_target"] for r in runs if r["steps_to_target"] is not None]
		rows.append((f"runs reaching target {target:g}", f"{len(reached)}/{len(runs)}"))
		if reached:
			rows.append(("env steps to reach target", _fmt(_mean_std(reached), 0)))
	return rows


def plot_learning_curve(runs: list[dict], experiment: str, out: Path) -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	grid_max = min(r["curve"][0][-1] for r in runs)
	grid_min = max(r["curve"][0][0] for r in runs)
	fig, ax = plt.subplots(figsize=(7, 4))

	if grid_max > grid_min:
		grid = np.linspace(grid_min, grid_max, 300)
		curves = np.stack(
			[np.interp(grid, r["curve"][0], r["curve"][1]) for r in runs]
		)
		mean, std = curves.mean(axis=0), curves.std(axis=0)
		for c in curves:
			ax.plot(grid, c, color="tab:blue", alpha=0.2, linewidth=0.8)
		ax.plot(grid, mean, color="tab:blue", label=f"mean of {len(runs)} runs")
		ax.fill_between(grid, mean - std, mean + std, color="tab:blue", alpha=0.2, label="± 1 std")
	else:  # a single very short run: plot it as is
		for r in runs:
			ax.plot(*r["curve"], label=r["name"])

	ax.set_xlabel("environment steps")
	ax.set_ylabel("episode reward (moving average)")
	ax.set_title(experiment)
	ax.grid(alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(out, dpi=150)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.split("\n")[1]) # pyright: ignore[reportOptionalMemberAccess]
	parser.add_argument("experiment")
	parser.add_argument("--runs-dir", default="runs")
	parser.add_argument("--results-dir", default="results")
	parser.add_argument("--window", type=int, default=20, help="episodes in the moving average")
	parser.add_argument("--target", type=float, default=None, help="reward threshold for steps-to-target")
	parser.add_argument("--include-unfinished", action="store_true", help="include failed/interrupted/running runs")
	args = parser.parse_args()

	exp_dir = Path(args.runs_dir) / args.experiment
	run_dirs = sorted(p for p in exp_dir.glob("*") if p.is_dir())
	runs = [r for p in run_dirs if (r := load_run(p, args.window, args.target))]
	skipped = [r["name"] for r in runs if r["status"] != "completed"]
	if not args.include_unfinished:
		runs = [r for r in runs if r["status"] == "completed"]
	if not runs:
		raise SystemExit(f"No usable runs found in {exp_dir}")

	out_dir = Path(args.results_dir) / args.experiment
	out_dir.mkdir(parents=True, exist_ok=True)

	table = summarize(runs, args.target)
	with open(out_dir / "summary.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["metric", "mean ± std over runs"])
		writer.writerows(table)

	lines = [f"# {args.experiment}", "", "| metric | mean ± std over runs |", "|---|---|"]
	lines += [f"| {k} | {v} |" for k, v in table]
	lines += ["", "## Runs", "", "| run | seed | final reward | eval mean | status | git dirty |", "|---|---|---|---|---|---|"]
	for r in runs:
		eval_mean = "n/a" if r["eval_mean"] is None else f"{r['eval_mean']:.1f}"
		lines.append(f"| {r['name']} | {r['seed']} | {r['final_reward']:.1f} | {eval_mean} | {r['status']} | {r['git_dirty']} |")
	if skipped and not args.include_unfinished:
		lines += ["", f"Excluded (not completed): {', '.join(skipped)}"]
	if any(r["git_dirty"] for r in runs):
		lines += ["", "Warning: some runs were made with uncommitted code changes."]
	(out_dir / "summary.md").write_text("\n".join(lines) + "\n")

	plot_learning_curve(runs, args.experiment, out_dir / "learning_curve.png")
	print("\n".join(lines))
	print(f"\nWrote {out_dir}/summary.csv, summary.md, learning_curve.png")


if __name__ == "__main__":
	main()