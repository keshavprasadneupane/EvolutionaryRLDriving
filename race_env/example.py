"""uv run python example.py [single|multi] [--no-walls]

P toggles manual driving (W/A/S/D). Multi mode: you drive car 0.
Tab = follow next car, L = lidar, wheel = zoom, Esc = quit.
"""
import sys

import numpy as np
import pygame

from race_env import MultiCarRace, SingleCarRace

multi = "multi" in sys.argv
kw = {"wall_thickness": 0.0, "grass_padding": 50.0} if "--no-walls" in sys.argv else {}
kw["ray_count"] = 11
env: SingleCarRace | MultiCarRace
if multi:
	env = MultiCarRace(render_mode="human", num_cars=8, auto_reset=True, **kw)
else:
	env = SingleCarRace(render_mode="human", **kw)

RISE, FALL = 2.5, 5.0  # per second, so the keys feel analog
DT = 1.0 / env.metadata["render_fps"]


def approach(cur: float, target: float, delta: float) -> float:
	if cur < target:
		return min(cur + delta, target)
	return max(cur - delta, target)


manual = False
p_prev = False
throttle = steer = 0.0

obs, info = env.reset(seed=42)
for _ in range(5000):
	# keys reflect the state at the last event pump (done inside env.render())
	keys = pygame.key.get_pressed()
	p_down = bool(keys[pygame.K_p])
	if p_down and not p_prev:
		manual = not manual
		throttle = steer = 0.0
		print("manual control:", "ON (W/A/S/D)" if manual else "OFF")
	p_prev = p_down

	action = env.action_space.sample()

	if manual:
		t_target = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
		s_target = float(keys[pygame.K_a]) - float(keys[pygame.K_d])  # +steer = left
		throttle = approach(throttle, t_target, (RISE if t_target else FALL) * DT)
		steer = approach(steer, s_target, (RISE if s_target else FALL) * DT)
		if multi:
			action[0] = (throttle, steer)  # car 0 is the player
		else:
			action = np.array([throttle, steer], dtype=np.float32)

	obs, reward, terminated, truncated, info = env.step(action)
	#print(f"\n Obs size: {obs.shape}\nObservation: {obs}\nReward: {reward}\nTerminated: {terminated}\nTruncated: {truncated}\nInfo: {info}\n\n")
	if not multi and (terminated or truncated):
		obs, info = env.reset()
	if env._renderer is not None and env._renderer.closed:
		break
env.close()