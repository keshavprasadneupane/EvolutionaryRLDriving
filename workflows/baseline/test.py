from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import threading
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from agents.dqn.agent import DQNetworkTestAgent
from environments.car_race import CarRacing
from network.convnet import ConvNet
from training.run_logger import RunLogger
from utility.basic import load_checkpoint, load_config
from .config import Config
from .train import (
	CURRENT_DIR,
	get_checkpoint_dir,
	get_final_model_dir,
)


DEFAULT_CONFIG_PATH = "config.json"

# Checkpoints written before the per-experiment layout lived in  checkpoints/
LEGACY_CHECKPOINT_DIR = CURRENT_DIR / "checkpoints"

# CarRacing-v3 with continuous=False: 0 noop, 1 steer left, 2 steer right, 3 gas, 4 brake
ACTION_NAMES = {0: "noop", 1: "left", 2: "right", 3: "gas", 4: "brake"}

EPISODE_PATTERN = re.compile(r"epi(\d+)")
SEED_PATTERN = re.compile(r"_seed(\d+)")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Evaluate a trained CarRacing DQN model (final model or a checkpoint). "
			"Results are appended to the eval.csv of the training run the model came "
			"from (found the same way train.py finds it when resuming) unless "
			"--no-record is given."
		)
	)

	parser.add_argument(
		"exp_name",
		type=str,
		help="Experiment name (required). Models are looked up under checkpoint/<exp_name>/ and models/<exp_name>/.",
	)
	parser.add_argument(
		"--model-type",
		choices=("final", "checkpoint"),
		default=None,
		help=(
			"Which model to load. Defaults to 'checkpoint' when --checkpoint-name is "
			"given, otherwise 'final' (also when --model-name is given)."
		),
	)
	parser.add_argument(
		"--checkpoint-name",
		type=str,
		default=None,
		help=(
			"Checkpoint file name (not a path), with or without .pth, e.g. "
			"'dqn_carracing_baseline_checkpoint_epi500_seed42'. A unique fragment such as "
			"'epi500' also works. If omitted, the most recent checkpoint of the "
			"experiment is used."
		),
	)
	parser.add_argument(
		"--model-name",
		type=str,
		default=None,
		help=(
			"Final model file name (not a path), with or without .pth, e.g. "
			"'dqn_carracing_baseline_final_seed42'. A unique fragment such as 'seed7' "
			"also works. If omitted, the most recent final model of the experiment is used."
		),
	)
	parser.add_argument(
		"--train-seed",
		type=int,
		default=None,
		help=(
			"Only consider models trained with this seed (the seed in the file name). "
			"If omitted, the newest file across all seeds is used."
		),
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=None,
		help="Explicit path to a model file. Overrides --model-type/--checkpoint-name/--model-name.",
	)
	parser.add_argument(
		"--no-record",
		action="store_true",
		help=(
			"Do not write anything to eval.csv (or the sweep/ file with --sweep). "
			"Results are only printed."
		),
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
		help="First evaluation seed (inclusive). Each seed is one evaluation episode.",
	)
	parser.add_argument(
		"--seed-max",
		type=int,
		default=10_004,
		help=(
			"Last evaluation seed (inclusive). Ignored with --render-mode human, "
			"which loops forever starting at --seed-min."
		),
	)
	parser.add_argument(
		"--render-mode",
		choices=("state_pixels", "human"),
		default="state_pixels",
		help=(
			"Use human for interactive rendering: episodes loop forever with a new seed "
			"each run starting at --seed-min, Enter skips to the next run, Ctrl+C quits. "
			"Human mode does not record results."
		),
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help="Override the configured device, e.g. cpu or cuda.",
	)
	parser.add_argument(
		"--debug-actions",
		action="store_true",
		help=(
			"Print the first actions/Q-values of each episode and a per-episode "
			"action histogram. Use this to detect a collapsed (constant-action) policy."
		),
	)

	parser.add_argument(
		"--eval-epsilon",
		type=float,
		default=0.0,
		help=(
			"Probability of replacing the greedy action with a random one, like "
			"training did. 0 (default) is purely greedy. Use e.g. 0.05 to check "
			"whether the policy only moves when exploration noise kicks it off."
		),
	)

	parser.add_argument(
		"--sweep",
		action="store_true",
		help=(
			"Greedily evaluate every checkpoint of the experiment on the same seeds "
			"(--seed-min/--seed-max) and print one row per checkpoint. Use it to "
			"find where a run's policy collapsed. Writes sweep/sweep_<sweep-name>.csv "
			"inside the training run directory (found from the checkpoints' "
			"metadata; next to this script if it can't be found) unless --no-record."
		),
	)
	parser.add_argument(
		"--sweep-name",
		type=str,
		default=None,
		help=(
			"With --sweep: unique name appended to the sweep file name, giving "
			"sweep/sweep_<sweep-name>.csv. Refuses to overwrite an existing file. "
			"If omitted, a timestamp is used."
		),
	)
	parser.add_argument(
		"--sweep-from",
		type=int,
		default=None,
		help="With --sweep: first checkpoint episode to include (e.g. 2200).",
	)
	parser.add_argument(
		"--sweep-to",
		type=int,
		default=None,
		help="With --sweep: last checkpoint episode to include (e.g. 2300).",
	)

	return parser.parse_args()


def resolve_path(path: Path) -> Path:
	"""Resolve relative paths against the project root."""
	if path.is_absolute():
		return path.resolve()
	return (CURRENT_DIR.parent.parent / path).resolve()


# --------------------------------------------------------------------------- #
# Model lookup
# --------------------------------------------------------------------------- #

def _sorted_by_recency(files: list[Path]) -> list[Path]:
	"""Oldest first, newest last. Ties are broken by name for determinism."""
	return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))


def _pick_by_name(
	files: list[Path],
	name: str | None,
	kind: str,
	exp_name: str,
	seed_note: str,
) -> Path:
	"""
	Choose one file from `files` (oldest first).

	No name -> the newest file. Otherwise an exact file-name match (with or
	without .pth) wins, then a fragment that matches exactly one file.
	"""
	if name is None:
		return files[-1]

	wanted = Path(name).name  # tolerate a pasted path
	stem = wanted[:-4] if wanted.endswith(".pth") else wanted

	exact = [p for p in files if p.stem == stem]
	if exact:
		return exact[-1]

	partial = [p for p in files if stem in p.stem]
	if len(partial) == 1:
		return partial[0]

	if not partial:
		recent = "\n  ".join(p.name for p in files[-20:])
		raise FileNotFoundError(
			f"No {kind} matching '{name}' for experiment '{exp_name}'{seed_note}. "
			f"Most recent {kind}s:\n  {recent}"
		)

	matches = "\n  ".join(p.name for p in partial[-20:])
	raise ValueError(
		f"'{name}' matches several {kind}s; give a longer name:\n  {matches}"
	)


def list_checkpoints(exp_name: str, train_seed: int | None) -> list[Path]:
	"""All checkpoints of an experiment, oldest first."""
	ckpt_dir = get_checkpoint_dir(exp_name)
	pattern = f"*_seed{train_seed}.pth" if train_seed is not None else "*.pth"
	files = list(ckpt_dir.glob(pattern)) if ckpt_dir.is_dir() else []

	if not files and train_seed is None and LEGACY_CHECKPOINT_DIR.is_dir():
		# Old runs have no seed in the file name, so they can't be seed-filtered.
		files = list(LEGACY_CHECKPOINT_DIR.glob(f"{exp_name}_checkpoint_*.pth"))
		if files:
			print(f"Note: using checkpoints from the legacy directory {LEGACY_CHECKPOINT_DIR}")

	return _sorted_by_recency(files)


def resolve_checkpoint(
	exp_name: str,
	checkpoint_name: str | None,
	train_seed: int | None,
) -> Path:
	"""Find a checkpoint by name, or the most recent one if no name is given."""
	files = list_checkpoints(exp_name, train_seed)
	seed_note = f" with seed {train_seed}" if train_seed is not None else ""

	if not files:
		raise FileNotFoundError(
			f"No checkpoints found for experiment '{exp_name}'{seed_note} in "
			f"{get_checkpoint_dir(exp_name)}."
		)

	return _pick_by_name(files, checkpoint_name, "checkpoint", exp_name, seed_note)


def list_final_models(
	config: Config,
	exp_name: str,
	train_seed: int | None,
) -> list[Path]:
	"""All final models of an experiment, oldest first."""
	model_dir = get_final_model_dir(config, exp_name)
	pattern = (
		f"{exp_name}_final_seed{train_seed}.pth"
		if train_seed is not None
		else f"{exp_name}_final*.pth"
	)
	files = list(model_dir.glob(pattern)) if model_dir.is_dir() else []

	if not files and train_seed is None:
		# Final models saved before the per-experiment layout.
		legacy = CURRENT_DIR / config.save_model_path / f"{exp_name}_final.pth"
		if legacy.is_file():
			print(f"Note: using legacy final model {legacy}")
			files = [legacy]

	return _sorted_by_recency(files)


def resolve_final_model(
	config: Config,
	exp_name: str,
	model_name: str | None,
	train_seed: int | None,
) -> Path:
	"""Find a final model by name, or the most recent one if no name is given."""
	files = list_final_models(config, exp_name, train_seed)
	seed_note = f" with seed {train_seed}" if train_seed is not None else ""

	if not files:
		raise FileNotFoundError(
			f"No final models found for experiment '{exp_name}'{seed_note} in "
			f"{get_final_model_dir(config, exp_name)}."
		)

	return _pick_by_name(files, model_name, "final model", exp_name, seed_note)


def select_model_path(args: argparse.Namespace, config: Config) -> Path:
	if args.model_path is not None:
		return resolve_path(args.model_path)

	if args.checkpoint_name is not None and args.model_name is not None:
		raise ValueError("Use either --checkpoint-name or --model-name, not both.")

	model_type = args.model_type or ("checkpoint" if args.checkpoint_name else "final")

	if model_type == "final":
		if args.checkpoint_name is not None:
			raise ValueError("--checkpoint-name can't be used with --model-type final.")
		return resolve_final_model(
			config, args.exp_name, args.model_name, args.train_seed
		)

	if args.model_name is not None:
		raise ValueError("--model-name can't be used with --model-type checkpoint.")
	return resolve_checkpoint(args.exp_name, args.checkpoint_name, args.train_seed)


# --------------------------------------------------------------------------- #
# Run directory / training position
# --------------------------------------------------------------------------- #

def read_meta(model_path: Path, device: str) -> dict[str, Any]:
	"""The `meta` block (run_dir, episodes, total_steps, ...) stored in a model file."""
	checkpoint = load_checkpoint(model_path, device)
	meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else None
	return meta if isinstance(meta, dict) else {}


def episode_from_path(path: Path) -> int | None:
	"""Training episode encoded in a checkpoint file name (…epi500…), if any."""
	match = EPISODE_PATTERN.search(path.stem)
	return int(match.group(1)) if match else None


def seed_from_path(path: Path) -> int | None:
	"""Training seed encoded in a model file name (…_seed42…), if any."""
	matches = SEED_PATTERN.findall(path.stem)
	return int(matches[-1]) if matches else None


def resolve_run_dir(
	meta: dict[str, Any],
	exp_name: str,
	model_path: Path,
	train_seed: int | None,
) -> Path | None:
	"""
	Find the training run a model came from, the way train.py finds it on resume:
	the run_dir stored in the model's meta. Models without it fall back to the
	newest run of the experiment (and training seed) under RunLogger's root.
	"""
	recorded = meta.get("run_dir")
	if recorded and Path(recorded).is_dir():
		return Path(recorded)

	seed = seed_from_path(model_path)
	if seed is None:
		seed = train_seed
	return RunLogger.find_run_dir(exp_name, seed)


def read_training_position(
	run_dir: Path | None,
	model_path: Path | None = None,
) -> tuple[int, int]:
	"""
	Fallback for models without metadata: read the training episode and
	environment-step count from metrics.csv.

	For a checkpoint whose name contains epiN, the last metrics row with
	episode <= N is used. For final models (or names without epiN) the last
	row is used.
	"""
	if run_dir is None:
		return 0, 0

	metrics_path = run_dir / "metrics.csv"
	if not metrics_path.is_file():
		raise FileNotFoundError(
			f"Could not find training metrics file: {metrics_path}"
		)

	target = episode_from_path(model_path) if model_path is not None else None

	with metrics_path.open("r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		last_row: dict[str, str] | None = None
		matched_row: dict[str, str] | None = None

		for row in reader:
			last_row = row
			if target is not None and int(float(row.get("episode") or 0)) <= target:
				matched_row = row

	chosen = matched_row if matched_row is not None else last_row

	if chosen is None:
		return 0, 0

	return (
		int(float(chosen.get("episode") or 0)),
		int(float(chosen.get("env_steps") or 0)),
	)


def training_position(
	meta: dict[str, Any],
	run_dir: Path,
	model_path: Path,
) -> tuple[int, int]:
	"""
	(episode, env_steps) of the evaluated model in its training run. These
	identify the model; they are not the evaluation episode's own numbers.
	"""
	episode = meta.get("episodes")
	env_steps = meta.get("total_steps")
	if episode is not None and env_steps is not None:
		return int(episode), int(env_steps)
	return read_training_position(run_dir, model_path)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def load_model(
	model_path: Path,
	config: Config,
	device: str,
) -> tuple[DQNetworkTestAgent, dict[str, Any]]:
	"""Build the evaluation network and load the saved DQN state. Returns (agent, meta)."""
	network = ConvNet(
		in_channels=config.in_channels,
		output_dim=5,
		hidden_dim=config.hidden_dim,
		device=device,
	)

	agent = DQNetworkTestAgent(network=network)

	checkpoint = load_checkpoint(model_path, device)

	if not isinstance(checkpoint, dict):
		raise TypeError(
			f"Expected a state dictionary in {model_path}, "
			f"got {type(checkpoint).__name__}."
		)

	# Works for both training checkpoints and final models: only the policy
	# network weights are read.
	agent.load_save_state(checkpoint)

	agent.network.eval()

	meta = checkpoint.get("meta")
	return agent, meta if isinstance(meta, dict) else {}


def make_env(config: Config, render_mode: str) -> gym.Env:
	"""
	Create one CarRacing environment.

	The training workflow uses CarRacing(skip=4, n_frames=config.in_channels).
	"""
	return CarRacing(
		render_mode=render_mode,
		skip=4,
		n_frames=config.in_channels,
	)


def evaluate_episode(
	env: gym.Env,
	agent: DQNetworkTestAgent,
	*,
	seed: int,
	debug_actions: bool = False,
	should_skip: Callable[[], bool] | None = None,
	epsilon: float = 0.0,
) -> tuple[float, int, bool]:
	"""
	Run one greedy evaluation episode.

	Returns (reward, length, skipped). If `should_skip` becomes true mid-episode,
	the episode ends early and `skipped` is True.

	In human mode Gymnasium's CarRacing renders inside env.step(), so no
	separate env.render() call is needed here.
	"""
	state, _ = env.reset(seed=seed)

	total_reward = 0.0
	length = 0
	action_counts: Counter[int] = Counter()
	skipped = False

	while True:
		if should_skip is not None and should_skip():
			skipped = True
			break

		# The test agent selects the best action (greedy).
		with torch.no_grad():
			action, q_value = agent.get_action(state)

		# Optional epsilon-greedy override; the default 0.0 keeps it purely greedy.
		if epsilon > 0.0 and random.random() < epsilon:
			action = random.randrange(env.action_space.n) # type: ignore

		action_counts[int(action)] += 1

		if debug_actions and length < 20:
			print(
				f"  step {length:3d} | action={action} "
				f"({ACTION_NAMES.get(int(action), '?')}) | q={q_value:.3f}"
			)

		state, reward, terminated, truncated, _ = env.step(int(action))

		total_reward += float(reward)
		length += 1

		if terminated or truncated:
			break

	if debug_actions:
		summary = ", ".join(
			f"{ACTION_NAMES.get(a, a)}={n}"
			for a, n in sorted(action_counts.items())
		)
		print(f"  actions: {summary}")

	return total_reward, length, skipped


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


def sweep_checkpoints(
	args: argparse.Namespace,
	config: Config,
	device: str,
) -> None:
	"""
	Evaluate every checkpoint of the experiment on the same seeds and print one
	row per checkpoint, so you can see where the policy got better or collapsed.

	"stuck" counts episodes that ran the full 250 decisions with reward < -80,
	which is what a car that never moves scores (about -93).

	The result is written to sweep/sweep_<name>.csv only when recording is
	enabled (no --no-record). The run directory is found from the newest swept
	checkpoint, like a normal evaluation.
	"""
	def episode_of(path: Path) -> int:
		episode = episode_from_path(path)
		return episode if episode is not None else -1

	files = sorted(list_checkpoints(args.exp_name, args.train_seed), key=episode_of)
	if args.sweep_from is not None:
		files = [p for p in files if episode_of(p) >= args.sweep_from]
	if args.sweep_to is not None:
		files = [p for p in files if episode_of(p) <= args.sweep_to]

	if not files:
		raise FileNotFoundError(
			f"No checkpoints to sweep for experiment '{args.exp_name}' "
			f"in {get_checkpoint_dir(args.exp_name)} (check --sweep-from/--sweep-to)."
		)

	# Resolved up front so a name clash fails before the (long) sweep starts.
	out_path = None
	if not args.no_record:
		newest = files[-1]
		run_dir = resolve_run_dir(
			read_meta(newest, "cpu"), args.exp_name, newest, args.train_seed
		)
		out_path = resolve_sweep_path(args, run_dir)

	seeds = list(range(args.seed_min, args.seed_max + 1))
	env = make_env(config, "none")
	rows: list[dict[str, float | int | str]] = []

	print(
		f"Sweeping {len(files)} checkpoints on seeds {seeds[0]}..{seeds[-1]} "
		f"(eval epsilon {args.eval_epsilon})\n"
	)
	print(f"{'episode':>8} {'mean':>9} {'std':>8} {'min':>9} {'max':>9} {'len':>6} {'stuck':>7}")

	try:
		for path in files:
			agent, _ = load_model(path, config, device)

			rewards: list[float] = []
			lengths: list[int] = []
			for seed in seeds:
				reward, length, _ = evaluate_episode(
					env, agent, seed=seed, epsilon=args.eval_epsilon
				)
				rewards.append(reward)
				lengths.append(length)

			reward_arr = np.asarray(rewards)
			length_arr = np.asarray(lengths)
			stuck = int(np.sum((reward_arr < -80) & (length_arr >= 250)))

			row = {
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
			rows.append(row)

			print(
				f"{row['episode']:>8} {row['mean_reward']:>9.1f} {row['std_reward']:>8.1f} "
				f"{row['min_reward']:>9.1f} {row['max_reward']:>9.1f} "
				f"{row['mean_length']:>6.0f} {stuck:>4}/{len(seeds)}",
				flush=True,
			)
	except KeyboardInterrupt:
		print("\nStopped early; saving the rows collected so far."
			if not args.no_record else "\nStopped early.")
	finally:
		env.close()
		if rows:
			if out_path is not None:
				out_path.parent.mkdir(parents=True, exist_ok=True)
				with out_path.open("w", newline="", encoding="utf-8") as file:
					writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
					writer.writeheader()
					writer.writerows(rows)
				print(f"\nSweep results written to: {out_path}")
			else:
				print("\nRecording disabled (--no-record): sweep file not written.")

			best = max(rows, key=lambda r: r["mean_reward"])
			print(
				f"Best checkpoint: {best['checkpoint']} "
				f"(mean reward {best['mean_reward']:.1f})"
			)


def run_interactive(
	env: gym.Env,
	agent: DQNetworkTestAgent,
	args: argparse.Namespace,
) -> None:
	"""
	Loop rendered episodes until Ctrl+C. Pressing Enter ends the current run and
	starts a new one. Each run uses the next consecutive seed.
	"""
	skip = threading.Event()

	def watch_enter() -> None:
		# One line from the terminal == one Enter press.
		for _ in sys.stdin:
			skip.set()

	threading.Thread(target=watch_enter, daemon=True).start()

	print("Interactive mode: runs loop automatically.")
	print("  Enter  -> skip to a new run")
	print("  Ctrl+C -> quit\n")

	rewards: list[float] = []
	run = 0

	try:
		while True:
			seed = args.seed_min + run
			skip.clear()

			reward, length, skipped = evaluate_episode(
				env,
				agent,
				seed=seed,
				debug_actions=args.debug_actions,
				should_skip=skip.is_set,
				epsilon=args.eval_epsilon,
			)

			run += 1
			status = " (skipped)" if skipped else ""
			print(
				f"Run {run} | seed={seed} | reward={reward:.2f} "
				f"| length={length}{status}"
			)

			if not skipped:
				rewards.append(reward)
	except KeyboardInterrupt:
		print("\nStopped.")
	finally:
		if rewards:
			print(
				f"Completed runs: {len(rewards)} | mean reward: {np.mean(rewards):.2f} "
				f"(skipped runs excluded)"
			)


def run_evaluation(
	*,
	agent: DQNetworkTestAgent,
	meta: dict[str, Any],
	config: Config,
	args: argparse.Namespace,
	model_path: Path,
	run_dir: Path | None,
) -> None:
	"""
	Run interactive testing or seeded evaluation episodes (one per seed in
	[seed_min, seed_max]).

	Seeded episodes are appended to <run_dir>/eval.csv unless --no-record is set.
	"""
	env = make_env(config, args.render_mode)

	try:
		if args.render_mode == "human":
			run_interactive(env, agent, args)
			return

		record = not args.no_record

		if record and run_dir is None:
			raise ValueError(
				"Could not find the training run for this model: its metadata has no "
				"run_dir, and no run folder for it exists under the runs root. Run from "
				"the same directory as training, or pass --no-record."
			)

		eval_path: Path | None = None
		logger: Any = nullcontext()
		episode = env_steps = 0

		if record:
			assert run_dir is not None
			episode, env_steps = training_position(meta, run_dir, model_path)
			eval_path = run_dir / "eval.csv"
			# Append-only: never truncates the training run's metrics/eval rows.
			logger = RunLogger.for_eval(run_dir)
			print(
				f"Recording to {eval_path} "
				f"(training position: episode={episode}, env_steps={env_steps})"
			)
		else:
			print("Recording disabled (--no-record): eval.csv will not be written.")

		seeds = range(args.seed_min, args.seed_max + 1)
		total_episodes = len(seeds)
		rewards: list[float] = []

		with logger as eval_log:
			for completed, seed in enumerate(seeds):
				reward, length, _ = evaluate_episode(
					env,
					agent,
					seed=seed,
					debug_actions=args.debug_actions,
					epsilon=args.eval_epsilon,
				)

				if eval_log is not None:
					eval_log.log_eval_episode(
						episode=episode,
						env_steps=env_steps,
						eval_seed=seed,
						reward=reward,
						length=length,
					)

				rewards.append(reward)

				print(
					f"[{completed + 1}/{total_episodes}] "
					f"seed={seed} reward={reward:.2f} length={length}"
				)

		if eval_path is not None:
			print(f"\nEvaluation complete. Results appended to: {eval_path}")
		else:
			print("\nEvaluation complete. Nothing was recorded.")
		print(f"Episodes evaluated: {len(rewards)}")
		print(f"Mean reward: {np.mean(rewards):.2f}")
		print(f"Std reward:  {np.std(rewards):.2f}")

	finally:
		env.close()


def main() -> None:
	args = parse_args()

	if args.seed_max < args.seed_min:
		raise ValueError("--seed-max must be >= --seed-min.")

	if args.sweep and args.render_mode == "human":
		raise ValueError("--sweep can't be combined with --render-mode human.")

	if not 0.0 <= args.eval_epsilon <= 1.0:
		raise ValueError("--eval-epsilon must be between 0 and 1.")

	if args.eval_epsilon > 0.0:
		note = (
			"Rows appended to eval.csv will not be comparable to greedy results."
			if not args.no_record
			else "Results are not being recorded."
		)
		print(f"Note: eval epsilon = {args.eval_epsilon} (not purely greedy). {note}")

	random.seed(args.seed_min)
	np.random.seed(args.seed_min)
	torch.manual_seed(args.seed_min)

	config_path = (
		resolve_path(args.config_path)
		if args.config_path is not None
		else CURRENT_DIR / DEFAULT_CONFIG_PATH
	)

	config = load_config(
		config_cls=Config,
		config_path=str(config_path),
		base_dir=CURRENT_DIR,
	)

	if config is None:
		raise FileNotFoundError(
			f"Could not load configuration from {config_path}. "
			"Create config.json first using the training workflow."
		)

	device = args.device or config.device

	if args.sweep:
		print(f"Experiment: {args.exp_name}")
		print(f"Device: {device}")
		sweep_checkpoints(args, config, device)
		return

	model_path = select_model_path(args, config)

	if not model_path.is_file():
		raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

	agent, meta = load_model(model_path, config, device)

	recording = not args.no_record and args.render_mode != "human"
	run_dir = (
		resolve_run_dir(meta, args.exp_name, model_path, args.train_seed)
		if recording
		else None
	)

	print(f"Experiment: {args.exp_name}")
	print(f"Model:  {model_path}")
	print(f"Device: {device}")
	print(f"Config: {config_path}")

	if run_dir is not None:
		print(f"Run:    {run_dir}")

	run_evaluation(
		agent=agent,
		meta=meta,
		config=config,
		args=args,
		model_path=model_path,
		run_dir=run_dir,
	)


if __name__ == "__main__":
	main()