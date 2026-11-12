uv run python -m pybind11_stubgen race_env._core -o . --enum-class-locations "DriveMode:race_env._core"
uv run python -c "
from race_env.envs import SingleCarRace
print('Checking the environment...')
env = SingleCarRace()
obs, info = env.reset(seed=0)
print(obs.shape)
print(env.step(env.action_space.sample())[1:4])
print('Done!')
"