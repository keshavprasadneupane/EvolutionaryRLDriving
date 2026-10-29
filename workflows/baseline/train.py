from pathlib import Path
import argparse
import os
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from utility.basic import (
	load_config,
	create_default_config,
	episode_to_reach_min_epsilon,
	load_checkpoint,
	save_the_model,
)
from training.run_logger import RunLogger
from agents.dqn.agent import DQNetworkTrainAgent
from network.convnet import ConvNet
from .config import Config
from environments.car_race import CarRacing

CURRENT_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = "config.json"

CHECKPOINT_ROOT = CURRENT_DIR / "checkpoint"
DEFAULT_EXP_NAME = "double_dqn"
DEFAULT_SEED = 42

SAVE_REPLAY = False
TRAIN_EVERY = 3  # train every N environment steps (not episodes). This is a hyperparameter that can be tuned for performance.
NUM_ENVS = 10    # Number of parallel CPU environments for AsyncVectorEnv

torch.backends.cudnn.benchmark = True


def get_checkpoint_dir(exp_name: str) -> Path:
	"""Directory holding every checkpoint of one experiment."""
	return CHECKPOINT_ROOT / exp_name


def get_checkpoint_filename(exp_name: str, episodes: int, seed: int) -> str:
	return f"{exp_name}_checkpoint_epi{episodes}_seed{seed}.pth"


def get_pointer_name(seed: int) -> str:
	# One pointer per seed, so runs with different seeds in the same experiment
	# directory never resume from each other's checkpoints.
	return f"current_checkpoint_seed{seed}.txt"


def get_final_model_dir(config: Config, exp_name: str) -> Path:
	return CURRENT_DIR / config.save_model_path / exp_name


def get_final_model_path(config: Config, exp_name: str, seed: int) -> Path:
	return get_final_model_dir(config, exp_name) / f"{exp_name}_final_seed{seed}.pth"


def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	random.seed(seed)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--fresh",
		action="store_true",
		help="Ignore any existing checkpoint and start training from scratch.",
	)
	parser.add_argument(
		"--exp-name",
		type=str,
		default=DEFAULT_EXP_NAME,
		help=(
			"Experiment name. Checkpoints go to checkpoint/<exp-name>/ and the "
			"final model to <save_model_path>/<exp-name>/."
		),
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=DEFAULT_SEED,
		help="Random seed. It is appended to every checkpoint/final-model filename.",
	)
	return parser.parse_args()


def make_env(config: Config, seed_offset: int):
	"""Factory function for vectorized environments."""
	def _thunk():
		return CarRacing(skip=4, n_frames=config.in_channels)
	return _thunk


def create_agent(config: Config, seed: int) -> DQNetworkTrainAgent:
	"""Creates and initializes the DQN training agent for CarRacing-v3."""
	network = ConvNet(
		in_channels=config.in_channels,
		output_dim=5,
		hidden_dim=config.hidden_dim,
		device=config.device,
	)
	optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

	return DQNetworkTrainAgent(
		network=network,
		optimizer=optimizer,
		target_update_freq=config.target_update_freq,
		tau=config.tau,
		replay_capacity=config.replay_capacity,
		batch_size=config.replay_batch_size,
		warmup=config.replay_warmup,
		double_dqn=True,
		seed=seed,
	)


class SkipGradientUpdate:
	"""Decides which env steps run a gradient update: one out of every `every` steps."""

	def __init__(self, every: int, start_step: int = 0) -> None:
		if every < 1:
			raise ValueError("every must be at least 1.")
		self.every = every
		self._step = start_step

	def should_update(self) -> bool:
		self._step += 1
		return self._step % self.every == 0


def _atomic_write_text(path: Path, text: str) -> None:
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(text)
	os.replace(tmp, path)


def save_checkpoint(
	agent: DQNetworkTrainAgent,
	config: Config,
	ckpt_dir: Path,
	exp_name: str,
	seed: int,
	completed_episodes: int,
	epsilon: float,
	total_steps: int,
	run_dir: Path,
) -> Path:
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	ckpt_path = ckpt_dir / get_checkpoint_filename(exp_name, completed_episodes, seed)

	save_state = agent.get_save_state(include_replay=SAVE_REPLAY)
	save_state["meta"] = {
		"episodes": int(completed_episodes),
		"epsilon": float(epsilon),
		"total_steps": int(total_steps),
		"run_dir": str(run_dir),
		"config": config.to_dict(),
	}

	tmp_path = ckpt_path.with_suffix(".pth.tmp")
	save_the_model(tmp_path, save_state)
	os.replace(tmp_path, ckpt_path)

	_atomic_write_text(ckpt_dir / get_pointer_name(seed), str(ckpt_path))
	return ckpt_path


def find_latest_checkpoint(ckpt_dir: Path, seed: int) -> Path | None:
	pointer = ckpt_dir / get_pointer_name(seed)
	if not pointer.exists():
		return None
	recorded = Path(pointer.read_text().strip())
	if recorded.exists():
		return recorded
	local = ckpt_dir / recorded.name
	return local if local.exists() else None


def load_resume_checkpoint(
	config: Config,
	ckpt_dir: Path,
	seed: int,
) -> tuple[Config, dict[str, Any] | None, Path | None]:
	latest = find_latest_checkpoint(ckpt_dir, seed)
	if latest is None:
		print(f"-> No checkpoint found in {ckpt_dir} for seed {seed}. Starting fresh.")
		return config, None, None

	try:
		checkpoint = load_checkpoint(latest, config.device)
	except Exception as e:
		raise RuntimeError(
			f"Could not load {latest} with weights_only=True. Run with --fresh."
		) from e

	meta = checkpoint.get("meta", {})
	saved_config = meta.get("config", {})

	can_resume, mismatches = config.can_resume_with(saved_config)
	if not can_resume:
		print(f"-> Critical configuration mismatch with {latest}: {mismatches}\n   Starting fresh.")
		return config, None, None

	return config.merged_with_checkpoint(meta), checkpoint, latest


def train(config: Config, exp_name: str, seed: int, fresh: bool = False) -> None:
	checkpoint: dict[str, Any] | None = None
	latest: Path | None = None
	ckpt_dir = get_checkpoint_dir(exp_name)

	print(f"-> Experiment: {exp_name} | seed: {seed}")
	print(f"-> Checkpoint directory: {ckpt_dir}")

	if fresh:
		print("-> --fresh specified. Starting fresh.")
	else:
		config, checkpoint, latest = load_resume_checkpoint(config, ckpt_dir, seed)

	if checkpoint is not None:
		print("\n\n-> Effective configuration after merging with checkpoint:")
		config.print_config()

	env = gym.vector.AsyncVectorEnv([
		make_env(config, i) for i in range(NUM_ENVS)
	])
	agent = create_agent(config, seed)

	current_epsilon = config.initial_epsilon
	total_steps = 0
	start_epi = 0
	resume_run_dir: Path | None = None

	if checkpoint is not None:
		try:
			agent.load_save_state(checkpoint)
		except RuntimeError as e:
			env.close()
			raise RuntimeError(f"{latest} does not match current network. Run with --fresh.") from e

		meta = checkpoint["meta"]
		start_epi = int(meta.get("episodes", 0))
		current_epsilon = float(meta.get("epsilon", config.initial_epsilon))
		total_steps = int(meta.get("total_steps", 0))
		run_dir_str = meta.get("run_dir")
		if run_dir_str and Path(run_dir_str).exists():
			resume_run_dir = Path(run_dir_str)
		print(f"-> Resuming from {latest} (episode {start_epi}, epsilon {current_epsilon:.4f})")

	if start_epi >= config.epochs:
		print(f"-> Checkpoint already at episode {start_epi} >= epochs ({config.epochs}).")
		env.close()
		return

	schedule = SkipGradientUpdate(TRAIN_EVERY, start_step=total_steps)
	print(f"-> Using AsyncVectorEnv with {NUM_ENVS} parallel environments.")
	print(f"-> Gradient update every {TRAIN_EVERY} env step(s).")

	completed_episodes = start_epi

	with RunLogger(
		exp_name,
		seed=seed,
		config=config.to_dict(),
		device=config.device,
		checkpoint_path=ckpt_dir,
		flush_every=config.log_interval,
		resume_dir=resume_run_dir,
		resume_from_episode=start_epi,
		resume_env_steps=total_steps,
	) as run:
		pbar: Any = None
		try:
			states, _ = env.reset(seed=[seed + i + start_epi * NUM_ENVS for i in range(NUM_ENVS)])
			env_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
			env_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
			
			# Track loss per environment independently
			env_running_loss = np.zeros(NUM_ENVS, dtype=np.float32)
			env_loss_steps = np.zeros(NUM_ENVS, dtype=np.int32)

			pbar = tqdm(
				initial=completed_episodes,
				total=config.epochs,
				desc="Training DQN",
				dynamic_ncols=True,
			)

			while completed_episodes < config.epochs:
				# Select actions for all parallel environments
				actions = np.empty(NUM_ENVS, dtype=np.int64)
				for i in range(NUM_ENVS):
					if random.random() < current_epsilon:
						actions[i] = env.single_action_space.sample()
					else:
						act, _ = agent.get_action(states[i])
						actions[i] = int(act)

				next_states, rewards, terminations, truncations, infos = env.step(actions)

				for i in range(NUM_ENVS):
					done = terminations[i] or truncations[i]

					# Extract correct next_state if auto-reset happened
					if done and "final_observation" in infos:
						next_state_i = infos["final_observation"][i]
					else:
						next_state_i = next_states[i]

					reward = float(rewards[i])
					env_rewards[i] += reward
					env_lengths[i] += 1
					total_steps += 1

					loss = agent.update_weights(
						state=states[i],
						action=int(actions[i]),
						reward=reward,
						next_state=next_state_i,
						done=bool(done),
						discount_factor=config.discount_factor,
						learn=schedule.should_update(),
					)

					if loss > 0.0:
						env_running_loss[i] += loss
						env_loss_steps[i] += 1

					if done:
						completed_episodes += 1
						current_epsilon = max(config.min_epsilon, current_epsilon * config.epsilon_decay)
						
						mean_loss = (env_running_loss[i] / env_loss_steps[i]) if env_loss_steps[i] > 0 else None

						run.log_episode(
							episode=completed_episodes - 1,
							env_steps=total_steps,
							reward=float(env_rewards[i]),
							length=int(env_lengths[i]),
							epsilon=current_epsilon,
							loss=mean_loss,
						)

						# Reset trackers for this specific environment slot
						env_rewards[i] = 0.0
						env_lengths[i] = 0
						env_running_loss[i] = 0.0
						env_loss_steps[i] = 0

						pbar.update(1)

						if completed_episodes % config.save_interval == 0:
							ckpt_path = save_checkpoint(
								agent, config, ckpt_dir, exp_name, seed,
								completed_episodes, current_epsilon, total_steps, run.dir,
							)
							print(f"\n-> Checkpoint saved to {ckpt_path}")

						if completed_episodes >= config.epochs:
							break

					states[i] = next_states[i]

		except KeyboardInterrupt:
			if completed_episodes > start_epi:
				ckpt_path = save_checkpoint(
					agent, config, ckpt_dir, exp_name, seed,
					completed_episodes, current_epsilon, total_steps, run.dir,
				)
				print(f"\n-> Interrupted. Checkpoint saved to {ckpt_path}")
			raise
		finally:
			env.close()
			if pbar is not None:
				pbar.close()

		if completed_episodes % config.save_interval != 0:
			save_checkpoint(
				agent, config, ckpt_dir, exp_name, seed,
				completed_episodes, current_epsilon, total_steps, run.dir,
			)

		final_model_path = get_final_model_path(config, exp_name, seed)
		final_model_path.parent.mkdir(parents=True, exist_ok=True)
		save_the_model(final_model_path, agent.get_save_state(include_replay=False))
		print(f"-> Training complete! Final model saved to {final_model_path}")


def main():
	args = parse_args()
	set_seed(args.seed)

	config = load_config(config_cls=Config, config_path=DEFAULT_CONFIG_PATH, base_dir=CURRENT_DIR)
	if config is None:
		default_config_path, _ = create_default_config(config_cls=Config, config_path=DEFAULT_CONFIG_PATH, base_dir=CURRENT_DIR)
		print(f"Default config created at {default_config_path}. Please edit and run again.")
		return

	config.print_config()
	print(f"Time to reach min epsilon: {episode_to_reach_min_epsilon(config.initial_epsilon, config.min_epsilon, config.epsilon_decay)} episodes")

	train(config, exp_name=args.exp_name, seed=args.seed, fresh=args.fresh)


if __name__ == "__main__":
	main()