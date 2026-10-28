from typing import Any, ClassVar
from dataclasses import dataclass
from utility.config_base import ConfigBase

@dataclass
class Config(ConfigBase):
	# Training & Execution Control
	epochs: int = 50_000
	target_update_freq: int = 1000

	# Model Architecture
	hidden_dim: int = 256

	# Optimization & RL Hyperparameters
	learning_rate: float = 0.0001
	discount_factor: float = 0.99
	tau: float = 0.80

	# Exploration (Epsilon-Greedy)
	initial_epsilon: float = 1.0
	min_epsilon: float = 0.05
	epsilon_decay: float = 0.9995

	# Experience Replay Configuration
	replay_capacity: int = 50_000
	replay_batch_size: int = 64
	replay_warmup: int = 2_000

	# Logging & Checkpointing
	log_interval: int = 10  # episodes between writes to metrics.csv
	save_interval: int = 50  # episodes between model checkpoints

	# Hardware Configuration
	device: str = "cpu"

	save_model_path: str = "models"

	CRITICAL_FIELDS: ClassVar[frozenset[str]] = frozenset(
		{"hidden_dim", "device"}
	)
	OVERWRITABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
		{"epochs"}
	)
	BOUNDS: ClassVar[dict[str, tuple[Any, Any]]] = {
		# Training & Execution Control
		"epochs": (1, None),
		"target_update_freq": (1, None),
		"hidden_dim": (1, None),

		# Optimization & RL Hyperparameters
		"learning_rate": (0.0, 1.0),
		"discount_factor": (0.0, 1.0),
		"tau": (0.0, 1.0),

		# Exploration (Epsilon-Greedy)
		"initial_epsilon": (0.0, 1.0),
		"min_epsilon": (0.0, 1.0),
		"epsilon_decay": (0.0, 1.0),

		# Experience Replay Configuration
		"replay_capacity": (1, None),
		"replay_batch_size": (1, None),
		"replay_warmup": (0, None),

		# Logging & Checkpointing
		"log_interval": (1, None),
		"save_interval": (1, None),
	}