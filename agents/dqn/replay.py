from typing import NamedTuple,Any
import numpy as np
import torch

class Transitions(NamedTuple):
	"""A batch of transitions. Unpacks like a plain 5-tuple."""

	states: np.ndarray
	actions: np.ndarray
	rewards: np.ndarray
	next_states: np.ndarray
	dones: np.ndarray


class ReplayBuffer:
	"""
	Fixed-size ring buffer of transitions stored in preallocated NumPy arrays.

	Compared with a deque of tuples, sampling is O(batch_size) instead of
	O(buffer_size) (a deque is slow to index in the middle), memory is
	contiguous, and checkpoints are a handful of arrays instead of 100k tuples.
	"""

	def __init__(self, capacity: int = 100_000, seed: int | None = None) -> None:
		if capacity <= 0:
			raise ValueError("Replay capacity must be positive.")

		self.capacity = capacity
		self._rng = np.random.default_rng(seed)
		self.clear()

	def clear(self) -> None:
		"""Removes all stored transitions."""
		self._states: np.ndarray | None = None
		self._next_states: np.ndarray | None = None
		self._actions = np.zeros(self.capacity, dtype=np.int64)
		self._rewards = np.zeros(self.capacity, dtype=np.float32)
		self._dones = np.zeros(self.capacity, dtype=np.float32)
		self._pos = 0
		self._size = 0

	def __len__(self) -> int:
		return self._size

	def _allocate(self, state_shape: tuple[int, ...]) -> None:
		self._states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
		self._next_states = np.zeros(
			(self.capacity, *state_shape), dtype=np.float32
		)

	def add(
		self,
		state: np.ndarray,
		action: int,
		reward: float,
		next_state: np.ndarray,
		done: bool,
	) -> None:
		"""Adds one transition, overwriting the oldest when full."""
		state = np.asarray(state, dtype=np.float32)
		next_state = np.asarray(next_state, dtype=np.float32)

		if self._states is None:
			self._allocate(state.shape)
		assert self._states is not None and self._next_states is not None

		if state.shape != self._states.shape[1:] or (
			next_state.shape != self._states.shape[1:]
		):
			raise ValueError(
				f"Expected state shape {self._states.shape[1:]}, got "
				f"{state.shape} and {next_state.shape}."
			)

		i = self._pos
		self._states[i] = state
		self._next_states[i] = next_state
		self._actions[i] = action
		self._rewards[i] = reward
		self._dones[i] = float(done)

		self._pos = (i + 1) % self.capacity
		self._size = min(self._size + 1, self.capacity)

	def sample(self, batch_size: int) -> Transitions:
		"""Samples a random mini-batch without replacement."""
		if batch_size <= 0:
			raise ValueError("Batch size must be positive.")

		if self._size < batch_size:
			raise ValueError("Not enough experiences to sample a batch.")

		assert self._states is not None and self._next_states is not None

		idx = self._rng.choice(self._size, size=batch_size, replace=False)

		return Transitions(
			self._states[idx],
			self._actions[idx],
			self._rewards[idx],
			self._next_states[idx],
			self._dones[idx],
		)

	def get_save_state(self) -> dict[str, Any]:
		"""Returns the buffer as arrays in chronological order (oldest first)."""
		if self._size == 0:
			return {"format": 2, "capacity": self.capacity, "size": 0}

		assert self._states is not None and self._next_states is not None

		start = (self._pos - self._size) % self.capacity
		idx = (start + np.arange(self._size)) % self.capacity

		# Tensors (not NumPy arrays) so checkpoints load with torch.load's
		# default weights_only=True.
		return {
			"format": 2,
			"capacity": self.capacity,
			"size": self._size,
			"states": torch.from_numpy(self._states[idx]),
			"actions": torch.from_numpy(self._actions[idx]),
			"rewards": torch.from_numpy(self._rewards[idx]),
			"next_states": torch.from_numpy(self._next_states[idx]),
			"dones": torch.from_numpy(self._dones[idx]),
		}

	def load_save_state(self, state: dict[str, Any]) -> None:
		"""
		Restores transitions from a checkpoint.
		"""
		self.clear()

		n = int(state.get("size", 0))
		if n == 0:
			return

		keep = min(n, self.capacity)
		recent = slice(n - keep, n)

		states = ReplayBuffer._to_numpy(state["states"])
		self._allocate(states.shape[1:])
		assert self._states is not None and self._next_states is not None

		self._states[:keep] = states[recent]
		self._next_states[:keep] = ReplayBuffer._to_numpy(state["next_states"])[recent]
		self._actions[:keep] = ReplayBuffer._to_numpy(state["actions"])[recent]
		self._rewards[:keep] = ReplayBuffer._to_numpy(state["rewards"])[recent]
		self._dones[:keep] = ReplayBuffer._to_numpy(state["dones"])[recent]

		self._size = keep
		self._pos = keep % self.capacity

	@staticmethod
	def _to_numpy(value: Any) -> np.ndarray:
		"""Converts a checkpoint field (tensor or array-like) to a NumPy array."""
		if isinstance(value, torch.Tensor):
			return value.detach().cpu().numpy()
		return np.asarray(value)