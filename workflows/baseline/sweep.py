from __future__ import annotations

import argparse
import csv
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .train import CURRENT_DIR, get_checkpoint_dir
from .test import (
	episode_from_path,
	evaluate_episode,
	evaluation_seeds,
	list_checkpoints,
	load_eval_config,
	load_model,
	make_env,
	read_meta,
	resolve_run_dir,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Greedily evaluate every checkpoint of an experiment on the same seeds "
			"and rank them with a weighted score (reward, consistency, failure rate). "
			"Use it to find where a run's policy collapsed. Writes "
			"sweep/sweep_<sweep-name>.csv inside the training run directory (found "
			"from the checkpoints' metadata; next to this script if it can't be "
			"found) unless --no-record."
		)
	)

	parser.add_argument(
		"exp_name",
		type=str,
		help="Experiment name (required). Checkpoints are looked up under checkpoint/<exp_name>/.",
	)
	parser.add_argument(
		"--train-seed",
		type=int,
		default=None,
		help=(
			"Only consider checkpoints trained with this seed (the seed in the file "
			"name). If omitted, all seeds are included."
		),
	)
	parser.add_argument(
		"--no-record",
		action="store_true",
		help="Do not write the sweep file. Results are only printed.",
	)
	parser.add_argument(
		"--config-path",
		type=Path,
		default=None,
		help="Path to the configuration JSON. Defaults to this workflow's config.json.",
	)
	parser.add_argument(
		"--seed-min",
		type=int,
		default=10_000,
		help=(
			"First evaluation seed. The seed increases by 1 for each episode "
			"(seed-min, seed-min+1, ...)."
		),
	)
	parser.add_argument(
		"--episodes",
		type=int,
		default=5,
		help=(
			"Number of evaluation episodes per checkpoint. Seeds start at "
			"--seed-min and increment automatically."
		),
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help="Override the configured device, e.g. cpu or cuda.",
	)
	parser.add_argument(
		"--eval-epsilon",
		type=float,
		default=0.0,
		help=(
			"Probability of replacing the greedy action with a random one. "
			"0 (default) is purely greedy."
		),
	)
	parser.add_argument(
		"--sweep-name",
		type=str,
		default=None,
		help=(
			"Unique name appended to the sweep file name, giving "
			"sweep/sweep_<sweep-name>.csv. Refuses to overwrite an existing file. "
			"If omitted, a timestamp is used."
		),
	)
	parser.add_argument(
		"--sweep-from",
		type=int,
		default=None,
		help="First checkpoint episode to include (e.g. 2200).",
	)
	parser.add_argument(
		"--sweep-to",
		type=int,
		default=None,
		help="Last checkpoint episode to include (e.g. 2300).",
	)

	return parser.parse_args()


def resolve_sweep_path(args: argparse.Namespace, run_dir: Path | None) -> Path:
	"""
	Output file for a sweep: <run_dir or script dir>/sweep/sweep_<name>.csv.

	The name comes from --sweep-name, or a timestamp if not given. An existing
	file is never overwritten.
	"""
	name = args.sweep_name or datetime.now().strftime("%Y%m%d_%H%M%S")
	# Keep the file name safe: letters, digits, dot, dash, underscore only.
	name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")

	if not name:
		raise ValueError("--sweep-name has no usable characters.")

	out_path = (run_dir or CURRENT_DIR) / "sweep" / f"sweep_{name}.csv"

	if out_path.exists():
		raise FileExistsError(
			f"Sweep file already exists: {out_path}. Choose a different --sweep-name."
		)

	return out_path


def checkpoint_score(row: dict) -> float:
	"""Calculate a weighted checkpoint-selection score (0-100)."""

	def reward_score(value: float) -> float:
		"""Convert a reward value to a normalized score in [0, 1]."""
		# Expected reward range: [-100, 1000].
		return float(np.clip((value + 100.0) / 1100.0, 0.0, 1.0))

	mean_score = reward_score(row["mean_reward"])
	min_score = reward_score(row["min_reward"])
	max_score = reward_score(row["max_reward"])

	stability_score = 1.0 - float(
		# 300 is a heuristic reference for scaling reward standard deviation.
		# A standard deviation of 300 or higher receives zero stability points.
		np.clip(row["std_reward"] / 300.0, 0.0, 1.0)
	)

	reliability_score = 1.0 - (
		row["stuck"] / max(row["episodes"], 1)
	)

	return 100.0 * (
		0.40 * mean_score
		+ 0.25 * min_score
		+ 0.15 * stability_score
		+ 0.15 * reliability_score
		+ 0.05 * max_score
	)


def sweep_checkpoints(
	args: argparse.Namespace,
	config: Config,
	device: str,
) -> None:
	"""
	Evaluate every checkpoint on identical seeds and rank them using
	a weighted score based on reward, consistency, and failure rate.

	The result is written to sweep/sweep_<name>.csv when recording
	is enabled.
	"""

	def episode_of(path: Path) -> int:
		episode = episode_from_path(path)
		return episode if episode is not None else -1

	files = sorted(
		list_checkpoints(args.exp_name, args.train_seed),
		key=episode_of,
	)

	if args.sweep_from is not None:
		files = [
			p for p in files
			if episode_of(p) >= args.sweep_from
		]

	if args.sweep_to is not None:
		files = [
			p for p in files
			if episode_of(p) <= args.sweep_to
		]

	if not files:
		raise FileNotFoundError(
			f"No checkpoints to sweep for experiment '{args.exp_name}' "
			f"in {get_checkpoint_dir(args.exp_name)} "
			f"(check --sweep-from/--sweep-to)."
		)

	# Resolve output path before starting the potentially long sweep.
	out_path = None

	if not args.no_record:
		newest = files[-1]
		run_dir = resolve_run_dir(
			read_meta(newest, "cpu"),
			args.exp_name,
			newest,
			args.train_seed,
		)
		out_path = resolve_sweep_path(args, run_dir)

	seeds = evaluation_seeds(args.seed_min, args.episodes)

	if not seeds:
		raise ValueError("Evaluation seed range is empty.")

	env = make_env(config, "state_pixels")
	rows: list[dict[str, float | int | str]] = []

	print(
		f"Sweeping {len(files)} checkpoints on "
		f"{len(seeds)} seeds ({seeds[0]}..{seeds[-1]}) "
		f"(eval epsilon {args.eval_epsilon})\n"
	)

	print(
		f"{'episode':>8} {'score':>8} {'mean':>9} "
		f"{'std':>8} {'min':>9} {'max':>9} {'stuck':>7} "
	)
	try:
		for path in files:
			agent, _ = load_model(path, config, device)

			rewards: list[float] = []
			lengths: list[int] = []

			for seed in seeds:
				reward, length, _ = evaluate_episode(
					env,
					agent,
					seed=seed,
					epsilon=args.eval_epsilon,
				)

				rewards.append(reward)
				lengths.append(length)

			reward_arr = np.asarray(rewards, dtype=np.float64)
			length_arr = np.asarray(lengths, dtype=np.int64)

			stuck = int(
				np.sum(
					(reward_arr < -80)
					& (length_arr >= 250)
				)
			)

			row: dict[str, float | int | str] = {
				"episode": episode_of(path),
				"checkpoint": path.name,
				"mean_reward": float(reward_arr.mean()),
				"std_reward": float(reward_arr.std()),
				"min_reward": float(reward_arr.min()),
				"max_reward": float(reward_arr.max()),
				"mean_length": float(length_arr.mean()),
				"stuck": stuck,
				"episodes": len(seeds),
			}

			row["score"] = checkpoint_score(row)
			rows.append(row)

			print(
				f"{row['episode']:>8} "
				f"{row['score']:>8.2f} "
				f"{row['mean_reward']:>9.1f} "
				f"{row['std_reward']:>8.1f} "
				f"{row['min_reward']:>9.1f} "
				f"{row['max_reward']:>9.1f} "
				f"{row['stuck']:>4}/{len(seeds)} ",
				flush=True,
			)

	except KeyboardInterrupt:
		print(
			"\nStopped early; saving collected results."
			if not args.no_record
			else "\nStopped early."
		)

	finally:
		env.close()

		if rows:
			if out_path is not None:
				out_path.parent.mkdir(
					parents=True,
					exist_ok=True,
				)

				with out_path.open(
					"w",
					newline="",
					encoding="utf-8",
				) as file:
					writer = csv.DictWriter(
						file,
						fieldnames=list(rows[0].keys()),
					)
					writer.writeheader()
					writer.writerows(rows)

				print(f"\nSweep results written to: {out_path}")

			else:
				print(
					"\nRecording disabled (--no-record): "
					"sweep file not written."
				)

			ranked = sorted(
				rows,
				key=lambda r: r["score"],
				reverse=True,
			)

			print("\nTop checkpoints:")

			for rank, row in enumerate(ranked[:5], start=1):
				print(
					f"{rank}. Episode {row['episode']} | "
					f"Score: {row['score']:.2f} | "
					f"Mean: {row['mean_reward']:.1f} | "
					f"Std: {row['std_reward']:.1f} | "
					f"Min: {row['min_reward']:.1f} | "
					f"Max: {row['max_reward']:.1f} | "
					f"Stuck: {row['stuck']}/{row['episodes']}"
				)

			best = ranked[0]

			print(
				f"\nBest checkpoint: {best['checkpoint']} "
				f"(score {best['score']:.2f})"
			)


def main() -> None:
	args = parse_args()

	if args.episodes < 1:
		raise ValueError("--episodes must be >= 1.")

	if not 0.0 <= args.eval_epsilon <= 1.0:
		raise ValueError("--eval-epsilon must be between 0 and 1.")

	random.seed(args.seed_min)
	np.random.seed(args.seed_min)
	torch.manual_seed(args.seed_min)

	config, _ = load_eval_config(args.config_path)
	device = args.device or config.device

	print(f"Experiment: {args.exp_name}")
	print(f"Device: {device}")
	sweep_checkpoints(args, config, device)


if __name__ == "__main__":
	main()