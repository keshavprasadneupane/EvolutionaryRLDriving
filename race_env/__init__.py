"""race_env: Box2D top-down racing engine (`_core`) + gymnasium environments."""
from race_env import _core as core
from .envs import DISCRETE_BITS, MultiCarRace, SingleCarRace

__all__ = ["core", "SingleCarRace", "MultiCarRace", "DISCRETE_BITS"]

try:  # gym.make("RaceSingle-v0") / gym.make("RaceMulti-v0")
    import gymnasium as _gym

    for _id, _entry in (("RaceSingle-v0", "race_env.envs:SingleCarRace"),
                        ("RaceMulti-v0", "race_env.envs:MultiCarRace")):
        if _id not in _gym.registry:
            _gym.register(id=_id, entry_point=_entry)
except Exception:  # gymnasium missing / registry quirks must not break import
    pass