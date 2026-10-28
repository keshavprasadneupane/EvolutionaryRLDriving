from typing import Any, Self

import numpy as np
import torch

from network.nnetwork import NNewtork
from .replay import ReplayBuffer, Transitions

def polyak_update(
	target: torch.nn.Module,
	source: torch.nn.Module,
	tau: float,
) -> None:
	"""
	Moves target parameters toward source parameters.

	target = (1 - tau) * target + tau * source

	tau == 1.0 is a hard copy. Buffers (e.g. batch-norm statistics) are always
	copied, matching Stable-Baselines3.
	"""
	with torch.no_grad():
		if tau >= 1.0:
			target.load_state_dict(source.state_dict())
		else:
			for t_param, s_param in zip(
				target.parameters(), source.parameters()
			):
				t_param.lerp_(s_param, tau)
			for t_buf, s_buf in zip(target.buffers(), source.buffers()):
				t_buf.copy_(s_buf)
	target.eval()


class DQNetworkAgent:
	"""
	Shared inference functionality for DQN agents.

	Supports continuous observations and discrete actions. Exploration (for
	example epsilon-greedy) is the caller's responsibility.
	"""

	def __new__(cls, *args: Any, **kwargs: Any) -> Self:
		if cls is DQNetworkAgent:
			raise TypeError(
				"DQNetworkAgent is a base class and cannot be instantiated "
				"directly."
			)
		return super().__new__(cls)

	def __init__(self, *, network: NNewtork) -> None:
		self.network = network
		self.device = network.device

	def get_action(self, state: np.ndarray) -> tuple[int, float]:
		"""
		Selects the greedy action and returns its estimated Q-value.

		`state` is a single, unbatched observation (e.g. (C, H, W)). A batch
		dimension is added here because layers like nn.Flatten assume dim 0
		is the batch.
		"""
		q_values = self.network.predict(state[np.newaxis, ...])[0]
		action = int(np.argmax(q_values))
		return action, float(q_values[action])


class DQNetworkTrainAgent(DQNetworkAgent):
	"""
	DQN training agent with optional experience replay.

	batch_size > 0 trains on random mini-batches from a replay buffer.
	batch_size == 0 disables replay and trains on each transition as it
	arrives, which matches SimpleDQNTrainAgent.

	Other options are standard and mostly follow Stable-Baselines3 and the
	PyTorch DQN tutorial:

	- target_update_freq / tau: every `target_update_freq` gradient steps the
	  target network moves toward the policy network by `tau`. tau=1.0 is the
	  classic hard copy; tau=0.005 with target_update_freq=1 is a soft update.
	- double_dqn: the policy network picks the next action and the target
	  network evaluates it, which reduces Q-value overestimation.
	- max_grad_norm: gradient-norm clipping (Stable-Baselines3 uses 10).
	  Pass None to disable.
	- warmup: minimum number of stored transitions before learning starts.
	"""

	def __init__(
		self,
		*,
		network: NNewtork,
		optimizer: torch.optim.Optimizer,
		target_update_freq: int = 1000,
		tau: float = 1.0,
		double_dqn: bool = False,
		max_grad_norm: float | None = 10.0,
		replay_capacity: int = 50_000,
		batch_size: int = 64,
		warmup: int = 0,
		seed: int | None = None,
	) -> None:
		if target_update_freq <= 0:
			raise ValueError("Target update frequency must be positive.")
		if not 0.0 < tau <= 1.0:
			raise ValueError("Tau must be in the range (0, 1].")
		if max_grad_norm is not None and max_grad_norm <= 0:
			raise ValueError("Max gradient norm must be positive or None.")
		if batch_size < 0:
			raise ValueError("Batch size cannot be negative.")
		if warmup < 0:
			raise ValueError("Warmup cannot be negative.")
		if batch_size > 0:
			if replay_capacity <= 0:
				raise ValueError(
					"Replay capacity must be positive when replay is enabled."
				)
			if max(batch_size, warmup) > replay_capacity:
				raise ValueError(
					"Replay capacity must be at least max(batch_size, warmup), "
					"otherwise learning never starts."
				)

		super().__init__(network=network)

		self.optimizer = optimizer
		self.target_update_freq = target_update_freq
		self.tau = tau
		self.double_dqn = double_dqn
		self.max_grad_norm = max_grad_norm
		self.batch_size = batch_size
		self.warmup = warmup

		self.target_network = self.network.deep_copy()
		self.steps = 0

		# A batch size of zero disables experience replay completely.
		self.replay = batch_size > 0
		self.replay_buffer: ReplayBuffer | None = (
			ReplayBuffer(capacity=replay_capacity, seed=seed)
			if self.replay
			else None
		)

		self.update_target_network(tau=1.0)

	@property
	def learning_rate(self) -> float:
		"""Current learning rate of the first optimizer parameter group."""
		return float(self.optimizer.param_groups[0]["lr"])

	@learning_rate.setter
	def learning_rate(self, value: float) -> None:
		for group in self.optimizer.param_groups:
			group["lr"] = value

	@property
	def can_learn(self) -> bool:
		"""True when the next update_weights call will do a gradient step."""
		if self.replay_buffer is None:
			return True
		return len(self.replay_buffer) >= max(self.batch_size, self.warmup)

	def update_target_network(self, tau: float | None = None) -> None:
		"""Moves the target network toward the policy network by `tau`."""
		polyak_update(
			self.target_network,
			self.network,
			self.tau if tau is None else tau,
		)

	def _predict_target(self, state: np.ndarray) -> np.ndarray:
		"""Computes Q-values for a single, unbatched state using the target network."""
		return self.target_network.predict(state[np.newaxis, ...])[0]

	@torch.no_grad()
	def _next_state_value(self, next_states: torch.Tensor) -> torch.Tensor:
		"""Bootstrapped value of the next states, shape (batch,)."""
		target_q: torch.Tensor = self.target_network(next_states)

		if self.double_dqn:
			best_actions = self.network(next_states).argmax(
				dim=1, keepdim=True
			)
			return target_q.gather(1, best_actions).squeeze(1)

		return target_q.max(dim=1).values

	def update_weights(
		self,
		state: np.ndarray,
		action: int,
		reward: float,
		next_state: np.ndarray,
		done: bool,
		discount_factor: float = 0.99,
	) -> float:
		"""
		Stores the transition and, once ready, performs one gradient update.

		batch_size > 0: Uses experience replay.
		batch_size == 0: Trains directly on the current transition.

		`done` should be True only for true terminal states, not for time-limit
		truncation; otherwise the agent learns that the final state has no
		future value.

		Returns the loss, or 0.0 while waiting for replay warmup (check
		`can_learn` to tell the two apart).
		"""
		if not 0.0 <= discount_factor <= 1.0:
			raise ValueError("Discount factor must be between 0 and 1.")

		if self.replay_buffer is not None:
			self.replay_buffer.add(state, action, reward, next_state, done)

			if not self.can_learn:
				return 0.0

			batch = self.replay_buffer.sample(self.batch_size)
		else:
			# Online learning: use only the current transition. States keep
			# their original dtype (e.g. uint8); _optimize converts to float32.
			batch = Transitions(
				states=np.asarray(state)[None],
				actions=np.asarray([action], dtype=np.int64),
				rewards=np.asarray([reward], dtype=np.float32),
				next_states=np.asarray(next_state)[None],
				dones=np.asarray([float(done)], dtype=np.float32),
			)

		return self._optimize(batch, discount_factor)

	def _optimize(self, batch: Transitions, discount_factor: float) -> float:
		"""Runs one gradient step on a batch and returns the loss."""
		# States may arrive as uint8 (0-255); the network scales them itself.
		states = torch.as_tensor(
			batch.states, dtype=torch.float32, device=self.device
		)
		actions = torch.as_tensor(
			batch.actions, dtype=torch.long, device=self.device
		)
		rewards = torch.as_tensor(
			batch.rewards, dtype=torch.float32, device=self.device
		)
		next_states = torch.as_tensor(
			batch.next_states, dtype=torch.float32, device=self.device
		)
		dones = torch.as_tensor(
			batch.dones, dtype=torch.float32, device=self.device
		)

		self.network.train()

		# Q-values for the actions actually taken.
		current_q = (
			self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
		)

		with torch.no_grad():
			next_value = self._next_state_value(next_states)
			target_q = rewards + discount_factor * (1.0 - dones) * next_value

		# Compute loss using the network's internal loss function, 
		# which supports custom loss functions passed as input.
		loss: torch.Tensor = self.network.loss(current_q, target_q)

		self.optimizer.zero_grad(set_to_none=True)
		loss.backward()

		if self.max_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(
				self.network.parameters(), self.max_grad_norm
			)

		self.optimizer.step()

		self.steps += 1

		if self.steps % self.target_update_freq == 0:
			self.update_target_network()

		return float(loss.item())

	def get_save_state(self, *, include_replay: bool = True) -> dict[str, Any]:
		"""
		Returns training state.

		Pass include_replay=False to skip the replay buffer, which keeps
		checkpoints small at the cost of restarting with an empty buffer.
		"""
		state: dict[str, Any] = {
			"network_state_dict": self.network.state_dict(),
			"target_network_state_dict": self.target_network.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"steps": self.steps,
			"batch_size": self.batch_size,
			"warmup": self.warmup,
			"replay_enabled": self.replay,
		}

		if include_replay and self.replay_buffer is not None:
			state["replay_buffer_state"] = self.replay_buffer.get_save_state()

		return state

	def load_save_state(self, state: dict[str, Any]) -> None:
		"""
		Restores training state.

		batch_size, warmup and replay capacity come from this agent's
		constructor, not from the checkpoint, so you can resume with different
		settings. A checkpoint without replay data leaves the buffer empty.
		"""
		self.network.load_state_dict(state["network_state_dict"])

		target_state = state.get("target_network_state_dict")
		if target_state is not None:
			self.target_network.load_state_dict(target_state)
		else:
			self.update_target_network(tau=1.0)
		self.target_network.eval()

		self.optimizer.load_state_dict(state["optimizer_state_dict"])
		self.steps = int(state["steps"])

		if self.replay_buffer is not None:
			self.replay_buffer.clear()
			replay_state = state.get("replay_buffer_state")
			if replay_state is not None:
				self.replay_buffer.load_save_state(replay_state)


class DQNetworkTestAgent(DQNetworkAgent):
	"""DQN inference agent: no replay memory, optimizer or target network."""

	def __init__(self, *, network: NNewtork) -> None:
		super().__init__(network=network)
		self.network.eval()

	def load_save_state(self, state: dict[str, Any]) -> None:
		"""Loads model weights for inference."""
		self.network.load_state_dict(state["network_state_dict"])
		self.network.eval()