from pathlib import Path
import gymnasium as gym
import torch
import random
from utility.basic import (
	load_config,
	create_default_config,
	episode_to_reach_min_epsilon,
	preprocess_observation,
)
from training.run_logger import RunLogger
from agents.dqn.agent import DQNetworkTrainAgent
from network.convnet import ConvNet
from .config import Config


CURRENT_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = "config.json"
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)


def create_agent(config: Config) -> DQNetworkTrainAgent:
	"""Creates and initializes the DQN training agent for CarRacing-v3."""
	network = ConvNet(
		in_channels=3,
		output_dim=5,
		hidden_dim=config.hidden_dim,
		device=config.device
	)
	optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

	agent = DQNetworkTrainAgent(
		network=network,
		optimizer=optimizer,
		target_update_freq=config.target_update_freq,
		tau=config.tau,
		replay_capacity=config.replay_capacity,
		batch_size=config.replay_batch_size,
		warmup=config.replay_warmup,
		seed=SEED,
	)

	return agent


def train(config: Config) -> None:
	env = gym.make("CarRacing-v3", continuous=False)
	agent = create_agent(config)

	current_epsilon = config.initial_epsilon
	total_steps = 0
	exp_name = "dqn_carracing_baseline"

	with RunLogger(
		exp_name,
		seed=SEED,
		config=config.to_dict(),
		device=config.device,
		checkpoint_path=CURRENT_DIR / "checkpoints",
		flush_every=config.log_interval,
	) as run:
		for epi in range(config.epochs):
			raw_state, _ = env.reset(seed=SEED + epi)
			state_obs = preprocess_observation(raw_state)

			terminated = False
			truncated = False
			total_reward = 0.0
			episode_length = 0
			running_loss = 0.0
			loss_steps = 0

			while not (terminated or truncated):
				if random.random() < current_epsilon:
					action = env.action_space.sample()
				else:
					action, _ = agent.get_action(state_obs)

				next_raw_state, reward, terminated, truncated, _ = env.step(action)
				next_state_obs = preprocess_observation(next_raw_state)

				reward = float(reward)
				total_reward += reward
				episode_length += 1
				total_steps += 1

				loss = agent.update_weights(
					state=state_obs,
					action=int(action),
					reward=reward,
					next_state=next_state_obs,
					done=bool(terminated),
					discount_factor=config.discount_factor,
				)

				if loss > 0.0:
					running_loss += loss
					loss_steps += 1

				state_obs = next_state_obs

			current_epsilon = max(config.min_epsilon, current_epsilon * config.epsilon_decay)
			mean_loss = (running_loss / loss_steps) if loss_steps > 0 else None

			run.log_episode(
				episode=epi,
				env_steps=total_steps,
				reward=total_reward,
				length=episode_length,
				epsilon=current_epsilon,
				loss=mean_loss,
			)

			print(f"Episode {epi} | Steps: {episode_length} | Reward: {total_reward:.2f} | Epsilon: {current_epsilon:.4f}")

			# Save interval checkpoint with custom name pattern: expname_checkpoint_epi{N}.pth
			if (epi + 1) % config.save_interval == 0:
				ckpt_filename = f"{exp_name}_checkpoint_epi{epi + 1}.pth"
				ckpt_filepath = run.checkpoint_path / ckpt_filename
				torch.save(agent.get_save_state(include_replay=False), ckpt_filepath)
				print(f"-> Checkpoint saved to {ckpt_filepath}")

		# Save final model to the designated model save path from config
		final_model_dir = CURRENT_DIR / config.save_model_path
		final_model_dir.mkdir(parents=True, exist_ok=True)
		final_model_path = final_model_dir / f"{exp_name}_final.pth"
		
		torch.save(agent.get_save_state(include_replay=False), final_model_path)
		print(f"-> Training complete! Final model saved to {final_model_path}")


def main():
	config = load_config(config_cls=Config, config_path=DEFAULT_CONFIG_PATH, base_dir=CURRENT_DIR)
	if config is None:
		default_config_path, _ = create_default_config(config_cls=Config, config_path=DEFAULT_CONFIG_PATH, base_dir=CURRENT_DIR)
		print(f"Default config created at {default_config_path}. Please edit the config file and run the script again.")
		return
	else:
		config.print_config()
		print(f"Time to reach min epsilon: {episode_to_reach_min_epsilon(config.initial_epsilon, config.min_epsilon, config.epsilon_decay)} episodes")

	train(config)


if __name__ == "__main__":
	main()