import gymnasium as gym
import numpy as np
from typing import Any
from utility.preprocessing import preprocess_observation, FrameStack


class CarRacing(gym.Env):
	"""
	A wrapper for CarRacing-v3 that integrates your modular preprocessing 
	and frame stacking utilities with action frame-skipping. 
	Fully compatible with Gymnasium's AsyncVectorEnv.
	"""

	def __init__(self, max_grass_frames: int = 50, skip: int = 4, n_frames: int = 4):
		self.env = gym.make("CarRacing-v3", continuous=False)
		self.max_grass_frames = max_grass_frames
		self.skip = skip
		
		self.action_space = self.env.action_space
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(n_frames, 84, 84), dtype=np.uint8
		)
		self.metadata = self.env.metadata
		
		self.frames = FrameStack(n_frames)
		self.frame_buf = np.empty((84, 84), dtype=np.uint8)

	def reset(
		self,
		*,
		seed: int | None = None,
		options: dict[str, Any] | None = None,
	):
		raw_state, info = self.env.reset(seed=seed, options=options)
		processed = preprocess_observation(raw_state, out=self.frame_buf)
		stacked_state = self.frames.reset(processed)
		return stacked_state, info

	def step(self, action):
		total_reward = 0.0
		terminated = False
		truncated = False
		info = {}
		
		processed = None
		for _ in range(self.skip):
			raw_state, reward, terminated, truncated, info = self.env.step(action)
			total_reward += float(reward)
			processed = preprocess_observation(raw_state, out=self.frame_buf)
			if terminated or truncated:
				break
				
		if processed is None:
			raise RuntimeError("No observation was produced; skip must be positive")

		stacked_state = self.frames.step(processed)
		return stacked_state, total_reward, terminated, truncated, info

	def render(self):
		return self.env.render()

	def close(self):
		self.env.close()