
#!/usr/bin/env bash
set -e

echo "Generating C++ Python type stubs..."
uv run python -m pybind11_stubgen race_env._core -o . \
    --enum-class-locations "DriveMode:race_env._core"

echo
echo "Checking the environment..."

uv run python - <<'PY'
from race_env.envs import SingleCarRace

env = SingleCarRace()

try:
    obs, info = env.reset(seed=0)
    print(f"Observation shape: {obs.shape}")

    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    print(f"Step successful: reward={reward:.3f}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print("Environment check passed!")

finally:
    env.close()
PY
