"""Gymnasium environments written in Python on top of the raw engine (`_core`).

The engine only simulates and returns data (`get_observation`, `get_static`).
Everything else - what the observation vector contains, rewards, termination,
truncation, action mapping - is decided HERE, so change it freely:
override `_vector`, `_judge` or `_info`, or just edit them.

	env = SingleCarRace(render_mode="human")
	obs, info = env.reset(seed=0)
	obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

MultiCarRace drives N cars in one world and returns batched rows (like a
vector env): obs (N, obs_dim), reward (N,), terminated (N,), truncated (N,).
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import _core as core
from .render import PygameRenderer

# discrete action -> key bits
DISCRETE_BITS = (
	0,                                    # 0 noop
	core.TDC_UP,                          # 1 gas
	core.TDC_UP | core.TDC_LEFT,          # 2 gas + left
	core.TDC_UP | core.TDC_RIGHT,         # 3 gas + right
	core.TDC_DOWN,                        # 4 brake / reverse
	core.TDC_LEFT,                        # 5 left
	core.TDC_RIGHT,                       # 6 right
	core.TDC_DOWN | core.TDC_LEFT,        # 7 brake + left
	core.TDC_DOWN | core.TDC_RIGHT,       # 8 brake + right
)

S = core.SurfaceType


class _Tracker:
	"""Per-car episode bookkeeping."""
	__slots__ = ("steps", "last_index", "max_cp", "done", "terminated", "truncated")

	def __init__(self, last_index: int = 0):
		self.steps: int = 0
		self.last_index: int = last_index
		self.max_cp: int = 0
		self.done: bool = False
		self.terminated: bool = False
		self.truncated: bool = False


class _RaceBase(gym.Env[Any, Any]):
	metadata = {"render_modes": ["human"], "render_fps": 60}

	def __init__(
		self,
		render_mode: Optional[str] = None,
		*,
		continuous: bool = True,
		ray_count: int = 21,
		ray_range: float = 50.0,
		max_steps: int = 1000,
		frame_skip: int = 1,
		target_laps: int = 1,
		new_track_on_reset: bool = True,
		# world overrides (None = keep whatever `world_config` / defaults say)
		world_config: Optional[core.WorldConfig] = None,
		car_config: Optional[core.CarConfig] = None,
		wall_thickness: Optional[float] = None,   # 0 = no walls, grass fills the map
		grass_padding: Optional[float] = None,    # used when there are no walls
		num_checkpoints: Optional[int] = None,
		cars_collide: Optional[bool] = None,
		# reward shaping
		progress_reward: float = 0.1,      # per centerline sample travelled
		checkpoint_reward: float = 1.0,
		grass_penalty: float = 0.05,
		step_penalty: float = 0.01,
		crash_penalty: float = 5.0,
		finish_bonus: float = 10.0,
		crash_margin: float = 4.0,         # walls: crash = centre this close to the wall
		# observation scaling
		velocity_scale: float = 100.0,
		angular_scale: float = 5.0,
	):
		if render_mode not in (None, "human"):
			raise ValueError(f"render_mode must be None or 'human', got {render_mode!r}")
		self.render_mode = render_mode
		self.continuous = continuous
		self.max_steps = max(1, max_steps)
		self.frame_skip = max(1, frame_skip)
		self.target_laps = target_laps
		self.new_track_on_reset = new_track_on_reset
		self.progress_reward, self.checkpoint_reward = progress_reward, checkpoint_reward
		self.grass_penalty, self.step_penalty = grass_penalty, step_penalty
		self.crash_penalty, self.finish_bonus = crash_penalty, finish_bonus
		self.crash_margin = crash_margin
		self.velocity_scale, self.angular_scale = velocity_scale, angular_scale

		self._world_cfg = (core.WorldConfig(world_config) if world_config is not None
						   else core.WorldConfig())
		if wall_thickness is not None:
			self._world_cfg.track.wall_thickness = wall_thickness
		if grass_padding is not None:
			self._world_cfg.track.grass_padding = grass_padding
		if num_checkpoints is not None:
			self._world_cfg.num_checkpoints = num_checkpoints
		if cars_collide is not None:
			self._world_cfg.cars_collide = cars_collide

		self._car_cfg = (core.CarConfig(car_config) if car_config is not None
						 else core.CarConfig())
		self._lidar_cfg = core.LidarConfig()
		self._lidar_cfg.ray_count = max(0, ray_count)
		self._lidar_cfg.range = ray_range

		self.obs_dim: int = 8 + self._lidar_cfg.ray_count
		self._mode = core.DriveMode.Continuous if continuous else core.DriveMode.Discrete

		self._world: Optional[core.RaceWorld] = None
		# empty until the first reset(), then filled from world.get_static()
		self._static: dict[str, Any] = {}
		self._renderer: Optional[PygameRenderer] = None
		self._last_obs: list[core.CarObservation] = []
		self._last_reward: Any = 0.0

	# ------------------------------------------------------------ spaces helpers
	def _single_spaces(self) -> tuple[spaces.Box, spaces.Space[Any]]:
		inf = np.inf
		obs = spaces.Box(-inf, inf, (self.obs_dim,), np.float32)
		act: spaces.Space[Any]
		if self.continuous:
			act = spaces.Box(-1.0, 1.0, (2,), np.float32)
		else:
			act = spaces.Discrete(len(DISCRETE_BITS))
		return obs, act

	# ------------------------------------------------------- engine <-> gym glue
	def _build_world(self) -> None:  # implemented by subclasses
		raise NotImplementedError

	def _require_world(self) -> core.RaceWorld:
		if self._world is None:
			raise RuntimeError("world not built, call reset() first")
		return self._world

	def _prepare_world(self, seed: Optional[int], options: Optional[dict[str, Any]]) -> None:
		"""Create the world once, then (re)start the episode via world.reset().

		The engine must be reset before it can be stepped, so every episode
		goes through world.reset(new_track, seed). The track seed is always
		drawn from self.np_random (seeded by gym.Env.reset(seed=...)), so a
		given env seed gives the same sequence of tracks.
		"""
		new_track = self.new_track_on_reset
		if options and "new_track" in options:
			new_track = bool(options["new_track"])

		first = self._world is None
		if first:
			self._build_world()
		world = self._require_world()

		# the first episode and any explicit seed always get a (seeded) new
		# track, otherwise the seed would have no effect
		if first or seed is not None:
			new_track = True

		world_seed = int(self.np_random.integers(1, 2**31 - 1))
		world.reset(new_track=new_track, seed=world_seed)
		if new_track or not self._static:
			self._static = world.get_static()

	# ---------------------------------------------- what YOU decide (python side)
	def _vector(self, o: core.CarObservation) -> np.ndarray:
		"""Observation vector for one car, from a CarObservation."""
		half = float(self._static["track_half_width"])
		lock = self._car_cfg.steer_lock_angle
		v = np.empty(self.obs_dim, np.float32)
		v[0] = o.local_velocity[0] / self.velocity_scale
		v[1] = o.local_velocity[1] / self.velocity_scale
		v[2] = o.angular_velocity / self.angular_scale
		v[3] = o.steer_angle / lock if lock > 0 else 0.0
		v[4] = o.traction
		v[5] = o.lateral_offset / half
		v[6] = o.heading_alignment
		v[7] = 1.0 if o.surface == S.Grass else 0.0
		v[8:] = o.rays
		return v

	def _judge(self, o: core.CarObservation, t: _Tracker) -> tuple[float, bool, bool]:
		"""(reward, terminated, truncated) for one car after a step."""
		st = self._static
		n = int(st["num_samples"])
		t.steps += 1
		d = (o.track_index - t.last_index) % n
		if d > n // 2:
			d -= n
		t.last_index = o.track_index

		reward = self.progress_reward * d
		new_cp = o.max_checkpoints - t.max_cp
		if new_cp > 0:
			reward += self.checkpoint_reward * new_cp
		t.max_cp = max(t.max_cp, o.max_checkpoints)
		if o.surface == S.Grass:
			reward -= self.grass_penalty
		reward -= self.step_penalty

		crashed = o.surface in (S.Wall, S.Outside)
		if not crashed and st["has_walls"]:
			wall_dist = st["track_half_width"] + st["grass_width"] - self.crash_margin
			crashed = abs(o.lateral_offset) >= wall_dist
		terminated = False
		if crashed:
			terminated = True
			reward -= self.crash_penalty
		elif self.target_laps > 0 and o.laps_completed >= self.target_laps:
			terminated = True
			reward += self.finish_bonus
		truncated = (not terminated) and t.steps >= self.max_steps
		t.terminated, t.truncated = terminated, truncated
		t.done = terminated or truncated
		return float(reward), terminated, truncated

	def _info(self, o: core.CarObservation, t: _Tracker) -> dict[str, Any]:
		return {
			"speed": o.speed,
			"lateral_offset": o.lateral_offset,
			"checkpoints": o.checkpoints_crossed,
			"laps": o.laps_completed,
			"surface": int(o.surface),
			"episode_step": t.steps,
		}

	# ------------------------------------------------------------------ render
	def _hud(self, index: int, o: core.CarObservation, t: _Tracker) -> list[str]:
		return [f"car {index}   step {t.steps}",
				f"speed {o.speed:6.1f}   surface {str(o.surface).split('.')[-1]}",
				f"checkpoints {o.checkpoints_crossed}   laps {o.laps_completed}"]

	def _cast(self, index: int) -> dict[str, Any]:
		raise NotImplementedError

	def _trackers_list(self) -> list[_Tracker]:
		raise NotImplementedError

	def _check_ready(self) -> None:
		if self._world is None or self._world.needs_reset:
			raise RuntimeError("call reset() before step()")

	def render(self) -> None:
		if self.render_mode != "human" or not self._last_obs:
			return None
		if self._renderer is None:
			self._renderer = PygameRenderer(
				self._car_cfg.body_vertices, fps=self.metadata["render_fps"],
				title=type(self).__name__)
		r = self._renderer
		if r.closed:
			return None
		trackers = self._trackers_list()
		f = min(r.focus, len(self._last_obs) - 1)
		r.draw(self._static, self._last_obs, self._hud(f, self._last_obs[f], trackers[f]),
			   cast_view=self._cast)
		return None

	def close(self) -> None:
		if self._renderer is not None:
			self._renderer.close()
			self._renderer = None


# ======================================================================== single
class SingleCarRace(_RaceBase):
	"""One car. Standard gymnasium shapes: obs (obs_dim,), scalar reward."""

	def __init__(self, render_mode: Optional[str] = None, **kwargs: Any):
		super().__init__(render_mode, **kwargs)
		self.observation_space, self.action_space = self._single_spaces()
		self._tracker = _Tracker()

	def _build_world(self) -> None:
		self._world = core.SingleCarWorld(self._world_cfg, self._car_cfg,
										  self._lidar_cfg, self._mode)

	def _single(self) -> core.SingleCarWorld:
		w = self._world
		if not isinstance(w, core.SingleCarWorld):
			raise RuntimeError("world not built, call reset() first")
		return w

	def _trackers_list(self) -> list[_Tracker]:
		return [self._tracker]

	def _cast(self, index: int) -> dict[str, Any]:
		return self._single().cast_view()

	def reset(self, *, seed: Optional[int] = None,
			  options: Optional[dict[str, Any]] = None) -> tuple[np.ndarray, dict[str, Any]]:
		super().reset(seed=seed)
		self._prepare_world(seed, options)
		o = self._single().get_observation()
		self._tracker = _Tracker(o.track_index)
		self._last_obs = [o]
		self.render()
		return self._vector(o), self._info(o, self._tracker)

	def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
		self._check_ready()
		w = self._single()
		t = self._tracker
		if t.done:  # frozen until reset
			if self.continuous:
				w.set_continuous_input(0.0, 0.0)
			else:
				w.set_discrete_input(0)
		elif self.continuous:
			a = np.asarray(action, dtype=np.float32).reshape(2)
			w.set_continuous_input(float(a[0]), float(a[1]))
		else:
			w.set_discrete_input(DISCRETE_BITS[int(action)])

		for _ in range(self.frame_skip):
			w.step()

		o = w.get_observation()
		if t.done:
			reward, terminated, truncated = 0.0, t.terminated, t.truncated
		else:
			reward, terminated, truncated = self._judge(o, t)
		self._last_obs = [o]
		self.render()
		return self._vector(o), reward, terminated, truncated, self._info(o, t)


# ========================================================================= multi
class MultiCarRace(_RaceBase):
	"""N cars in one world, batched rows.

	obs (N, obs_dim) | action (N, 2) or MultiDiscrete([9]*N)
	reward (N,) | terminated (N,) | truncated (N,)
	auto_reset=True respawns a finished car on the next step (its row then
	carries a fresh start observation, reward 0, flags False).
	"""

	def __init__(self, render_mode: Optional[str] = None, *, num_cars: int = 8,
				 auto_reset: bool = False, **kwargs: Any):
		super().__init__(render_mode, **kwargs)
		if num_cars < 1:
			raise ValueError("num_cars must be >= 1")
		self.num_cars = num_cars
		self.auto_reset = auto_reset
		self.single_observation_space, self.single_action_space = self._single_spaces()
		inf = np.inf
		self.observation_space = spaces.Box(-inf, inf, (num_cars, self.obs_dim), np.float32)
		if self.continuous:
			self.action_space = spaces.Box(-1.0, 1.0, (num_cars, 2), np.float32)
		else:
			self.action_space = spaces.MultiDiscrete([len(DISCRETE_BITS)] * num_cars)
		self._ids: list[int] = []
		self._trackers: list[_Tracker] = []

	def _build_world(self) -> None:
		w = core.MultiCarWorld(self._world_cfg)
		spec = core.CarSpec()
		spec.car, spec.lidar, spec.mode = self._car_cfg, self._lidar_cfg, self._mode
		for _ in range(self.num_cars):
			w.add_car(spec)
		self._world = w
		self._ids = w.ids()

	def _multi(self) -> core.MultiCarWorld:
		w = self._world
		if not isinstance(w, core.MultiCarWorld):
			raise RuntimeError("world not built, call reset() first")
		return w

	def _trackers_list(self) -> list[_Tracker]:
		return self._trackers

	def _cast(self, index: int) -> dict[str, Any]:
		return self._multi().cast_view(self._ids[index])

	def _stack(self, obs_list: list[core.CarObservation]) -> np.ndarray:
		return np.stack([self._vector(o) for o in obs_list])

	def reset(self, *, seed: Optional[int] = None,
			  options: Optional[dict[str, Any]] = None) -> tuple[np.ndarray, dict[str, Any]]:
		super().reset(seed=seed)
		self._prepare_world(seed, options)
		obs = self._multi().get_observations()
		self._trackers = [_Tracker(o.track_index) for o in obs]
		self._last_obs = obs
		self.render()
		infos = [self._info(o, t) for o, t in zip(obs, self._trackers)]
		info = {k: np.array([d[k] for d in infos]) for k in infos[0]}
		return self._stack(obs), info

	# batched return types and the `actions` name differ from gym.Env.step on purpose
	def step(  # pyright: ignore[reportIncompatibleMethodOverride]
		self, actions: Any
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
		self._check_ready()
		w = self._multi()
		n, ids, trackers = self.num_cars, self._ids, self._trackers
		respawned = np.zeros(n, bool)
		if self.auto_reset:
			for i, t in enumerate(trackers):
				if t.done:
					w.reset_car(ids[i])
					trackers[i] = _Tracker()
					respawned[i] = True

		done = np.array([t.done for t in trackers])
		if self.continuous:
			a = np.asarray(actions, dtype=np.float32).reshape(n, 2).copy()
			a[done] = 0.0
			w.set_continuous_inputs(ids, a)
		else:
			d = np.asarray(actions).reshape(n)
			bits = [0 if done[i] else DISCRETE_BITS[int(d[i])] for i in range(n)]
			w.set_discrete_inputs(ids, bits)

		for _ in range(self.frame_skip):
			w.step()

		obs = w.get_observations()
		reward = np.zeros(n, np.float32)
		terminated = np.zeros(n, bool)
		truncated = np.zeros(n, bool)
		for i, (o, t) in enumerate(zip(obs, trackers)):
			if respawned[i]:
				t.last_index = o.track_index
			elif t.done:
				terminated[i], truncated[i] = t.terminated, t.truncated
			else:
				reward[i], terminated[i], truncated[i] = self._judge(o, t)
		self._last_obs = obs
		self.render()
		infos = [self._info(o, t) for o, t in zip(obs, trackers)]
		info = {k: np.array([d[k] for d in infos]) for k in infos[0]}
		return self._stack(obs), reward, terminated, truncated, info