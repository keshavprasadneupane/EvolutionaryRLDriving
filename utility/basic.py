from __future__ import annotations
from collections import deque
import dataclasses
from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING, Type, TypeVar, Any
import numpy as np
import torch as tch

if TYPE_CHECKING: # satisfies mypy's requirement for a type variable bound to a dataclass instance
	from _typeshed import DataclassInstance
	T = TypeVar("T", bound=DataclassInstance)
else:
	T = TypeVar("T")

def resolve_path(current_dir: Path, path: str | Path,) -> Path:
	"""Resolve a path relative to the current directory."""

	path = Path(path) if isinstance(path, str) else path
	
	if not path.is_absolute():
		path = current_dir / path

	return path.resolve()


def episode_to_reach_min_epsilon(
	initial_epsilon: float, 
	target_epsilon: float, 
	decay_rate: float
) -> int:
	"""Calculate episodes needed to reach the target minimum epsilon.

	Args:
		target_epsilon: The final target exploration rate (e.g., min_epsilon).
		initial_epsilon: The starting exploration rate (e.g., initial_epsilon).
		decay_rate: The multiplicative decay factor applied per episode (e.g., epsilon_decay).

	Returns:
		int: The number of episodes required to reach target_epsilon via 
		exponential decay (calculated as ceil(ln(target / initial) / ln(decay))).
	"""
	return int(np.ceil(np.log(target_epsilon / initial_epsilon) / np.log(decay_rate)))


def load_config(config_cls: Type[T], config_path: str, base_dir: Path) -> T | None:
	"""Load a JSON configuration file into the specified dataclass, returning None if missing."""
	resolved_path = resolve_path(base_dir, config_path)
	if not resolved_path.exists():
		return None

	with resolved_path.open("r", encoding="utf-8") as file:
		data = json.load(file)

	if not isinstance(data, dict):
		raise ValueError(f"Configuration must contain a JSON object: {resolved_path}")

	valid_keys = {item.name for item in dataclasses.fields(config_cls)}
	filtered_data = {key: value for key, value in data.items() if key in valid_keys}
	return config_cls(**filtered_data)


def create_default_config(config_cls: Type[T], config_path: str, base_dir: Path) -> tuple[Path, T]:
	"""Create a default JSON configuration file for the specified dataclass."""
	resolved_path = resolve_path(base_dir, config_path)
	resolved_path.parent.mkdir(parents=True, exist_ok=True)
	config_instance = config_cls()
	with resolved_path.open("w", encoding="utf-8") as file:
		json.dump(asdict(config_instance), file, indent=4)
	return resolved_path , config_instance



def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
	"""Load a checkpoint safely using weights_only=True."""
	loaded = tch.load(path, map_location=device, weights_only=True)

	if not isinstance(loaded, dict):
		raise ValueError("Invalid checkpoint format.")
	return loaded



def _to_safe_checkpoint_value(value: Any) -> Any:
	"""
	Convert NumPy values in replay memory and metadata to safe primitive values.

	This allows checkpoints to be loaded with torch.load(weights_only=True).
	Tensors and ordinary Python values are retained.
	"""
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, dict):
		return {key: _to_safe_checkpoint_value(item) for key, item in value.items()}
	if isinstance(value, (list, tuple, deque)):
		return [_to_safe_checkpoint_value(item) for item in value]
	return value




def save_the_model(
	save_path: Path,
	save_data: dict[str, Any],
) -> None:
	"""
	Save the model and metadata to a checkpoint file, converting NumPy values to safe primitive values.
	"""
	tch.save(_to_safe_checkpoint_value(save_data), save_path)
