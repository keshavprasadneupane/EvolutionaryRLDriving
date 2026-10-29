from typing import Any, ClassVar
from dataclasses import dataclass
from utility.config_base import ConfigBase

@dataclass
class Config(ConfigBase):
	# Experiment
	exp_name: str = "double_dqn"  # checkpoints -> checkpoint/<exp_name>/, final model -> <save_model_path>/<exp_name>/

	# Training & Execution Control
	epochs: int = 10_000
	target_update_freq: int = 500

	# Model Architecture
	in_channels: int = 4  # N stacked grayscale frames (1 = single frame)
	hidden_dim: int = 224 

	# Optimization & RL Hyperparameters
	learning_rate: float = 0.00025
	discount_factor: float = 0.99
	tau: float = 1.0

	# Exploration (Epsilon-Greedy)
	initial_epsilon: float = 1.0
	min_epsilon: float = 0.05
	epsilon_decay: float = 0.998

	# Experience Replay Configuration
	replay_capacity: int = 50_000
	replay_batch_size: int = 32
	replay_warmup: int = 100
	
	# Logging & Checkpointing
	log_interval: int = 10  # episodes between writes to metrics.csv
	save_interval: int = 50  # episodes between model checkpoints

	# Hardware Configuration
	device: str = "cuda"

	save_model_path: str = "models"

	CRITICAL_FIELDS: ClassVar[frozenset[str]] = frozenset(
		{"in_channels", "hidden_dim", }
	)
	OVERWRITABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
		{"exp_name", "epochs", "log_interval", "save_interval", "device", "save_model_path"}
	)
	BOUNDS: ClassVar[dict[str, tuple[Any, Any]]] = {
		"epochs": (1, None),
		"target_update_freq": (1, None),
		"in_channels": (1, None),
		"hidden_dim": (1, None),

		"learning_rate": (0.0, 1.0),
		"discount_factor": (0.0, 1.0),
		"tau": (0.0, 1.0),

		"initial_epsilon": (0.0, 1.0),
		"min_epsilon": (0.0, 1.0),
		"epsilon_decay": (0.0, 1.0),

		"replay_capacity": (1, None),
		"replay_batch_size": (1, None),
		"replay_warmup": (0, None),

		"log_interval": (1, None),
		"save_interval": (1, None),
	}