from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Self
import warnings
from enum import Enum

class MergeStrategy(str,Enum):
	OVERWRITE = "overwrite"
	UNION = "union"

@dataclass
class ConfigBase:
	"""Base configuration with validation and checkpoint compatibility support.

	Example:
		>>> @dataclass
		... class TrainingConfig(ConfigBase):
		...     episodes: int = 10_000
		...     learning_rate: float = 0.001
		...     device: str = "cpu"
		...
		...     CRITICAL_FIELDS: ClassVar[frozenset[str]] = frozenset(
		...         {"learning_rate", "device"}
		...     )
		...     OVERWRITABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
		...         {"episodes"}
		...     )
		...     BOUNDS: ClassVar[dict[str, tuple[Any, Any]]] = {
		...         "episodes": (1, None),
		...         "learning_rate": (0.0, 1.0),
		...     }
		...
		>>> config = TrainingConfig(episodes=20_000)
		>>> checkpoint = {"config": {
		...     "episodes": 10_000,
		...     "learning_rate": 0.001,
		...     "device": "cpu",
		... }}
		>>> can_resume, mismatches = config.can_resume_with(checkpoint["config"])
		>>> can_resume
		True
		>>> config = config.merged_with_checkpoint(checkpoint)
		>>> config.episodes
		20000

	Critical fields must match the saved configuration to resume safely.
	Overwritable fields retain their current values, while other fields use
	checkpoint values when available. BOUNDS defines inclusive minimum and
	maximum values; use None for an unbounded limit.
	"""

	CRITICAL_FIELDS: ClassVar[frozenset[str]] = frozenset()
	OVERWRITABLE_FIELDS: ClassVar[frozenset[str]] = frozenset()
	BOUNDS: ClassVar[dict[str, tuple[Any, Any]]] = {}

	def __post_init__(self) -> None:
		self._validate_rules()
		self._validate_bounds()

	def _validate_bounds(self) -> None:
		for name, (minimum, maximum) in self.BOUNDS.items():
			value = getattr(self, name, None)

			if value is None:
				continue
			if minimum is not None and value < minimum:
				raise ValueError(f"{name}={value} is below minimum {minimum}.")
			if maximum is not None and value > maximum:
				raise ValueError(f"{name}={value} exceeds maximum {maximum}.")

	def _validate_rules(self) -> None:
		overlaps = self.CRITICAL_FIELDS & self.OVERWRITABLE_FIELDS

		if overlaps:
			warnings.warn(
				f"Critical fields also marked overwritable: {sorted(overlaps)}. "
				"Critical fields take precedence.",
				UserWarning,
				stacklevel=2,
			)

	def can_resume_with(
		self, saved_config: dict[str, Any]
	) -> tuple[bool, list[str]]:
		mismatches = sorted(
			name
			for name in self.CRITICAL_FIELDS
			if name in saved_config
			and saved_config[name] != getattr(self, name)
		)
		return not mismatches, mismatches


	def merged_with_checkpoint(self, meta_data: dict[str, Any]) -> Self:
		saved_config = meta_data.get("config", {})

		if not isinstance(saved_config, dict):
			raise ValueError("Checkpoint configuration must be a dictionary.")

		current = asdict(self)

		merged = {
			name: (
				value
				if name in self.CRITICAL_FIELDS
				or name in self.OVERWRITABLE_FIELDS
				else saved_config.get(name, value)
			)
			for name, value in current.items()
		}

		return type(self)(**merged)

	def print_config(self):
		print("Current Configuration:")
		for field_name, field_value in self.to_dict().items():
			print(f"{field_name}: {field_value}")


	def to_dict(self) -> dict[str, Any]:
		"""Convert the configuration to a dictionary, optionally ignoring specified fields.

		Note:
			Subclasses can override `to_dict` to customize the output format. 
			Leave `_to_dict` intact to inherit proper handling of `ignore_fields` 
			and `overwrite_option`.
		"""
		return self._to_dict()

	def _to_dict(
		self, 
		ignore_fields: set[str] | None = None, 
		overwrite_option: MergeStrategy = MergeStrategy.UNION
	) -> dict[str, Any]:
		"""Internal helper for dictionary conversion with filtering strategy support."""
		super_ignore_field = {"CRITICAL_FIELDS", "OVERWRITABLE_FIELDS", "BOUNDS"}

		if ignore_fields is None:
			ignore_fields = super_ignore_field
		elif overwrite_option == MergeStrategy.OVERWRITE:
			# ignore_fields completely replaces the default super_ignore_field
			pass
		else:  # MergeStrategy.UNION
			ignore_fields = ignore_fields.union(super_ignore_field)

		return {
			field: getattr(self, field)
			for field in self.__dataclass_fields__
			if field not in ignore_fields
		}