"""Strip the replay buffer out of a saved checkpoint / model file.

Checkpoints written with SAVE_REPLAY = True can be huge. This script loads a
.pth artifact, drops the replay-buffer entries, and writes the slimmed file.
Network / optimizer / target-network state and the "meta" block are untouched,
so the result still resumes training (the buffer just starts empty and refills
during warmup).

Usage:
	python replay_remover.py path/to/checkpoint.pth
	python replay_remover.py path/to/checkpoint.pth -o slim.pth
	python replay_remover.py path/to/checkpoint.pth --inplace
	python replay_remover.py path/to/checkpoint.pth --dry-run
	python replay_remover.py path/to/checkpoint.pth --keys replay_buffer memory
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

# Top-level keys containing any of these substrings (case-insensitive) are treated as replay data.
REPLAY_HINTS = ("replay",)
# Never touched: "meta" holds the config, which has keys like replay_capacity needed for resuming.
PROTECTED_KEYS = {"meta"}


def human_size(num_bytes: int) -> str:
	size = float(num_bytes)
	for unit in ("B", "KB", "MB", "GB", "TB"):
		if size < 1024 or unit == "TB":
			return f"{size:.2f} {unit}"
		size /= 1024
	return f"{size:.2f} TB"


def load_artifact(path: Path, allow_unsafe: bool) -> Any:
	try:
		return torch.load(path, map_location="cpu", weights_only=True)
	except Exception as e:
		if not allow_unsafe:
			raise RuntimeError(
				f"Could not load {path} with weights_only=True ({e}).\n"
				"If you trust this file, re-run with --unsafe-load."
			) from e
		print("-> weights_only=True failed; falling back to weights_only=False (only do this for files you trust).")
		return torch.load(path, map_location="cpu", weights_only=False)


def find_replay_keys(state: dict, explicit_keys: list[str] | None) -> list[str]:
	if explicit_keys:
		return [k for k in explicit_keys if k in state and k not in PROTECTED_KEYS]
	return [
		k for k in state
		if k not in PROTECTED_KEYS
		and any(hint in str(k).lower() for hint in REPLAY_HINTS)
	]


def default_output_path(src: Path) -> Path:
	return src.with_name(f"{src.stem}_noreplay{src.suffix}")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Remove the replay buffer from a checkpoint or model file.")
	p.add_argument("path", type=Path, help="Path to the checkpoint / model artifact (.pth).")
	p.add_argument("-o", "--output", type=Path, default=None,
		help="Where to write the result. Default: <name>_noreplay<ext> next to the input.")
	p.add_argument("--inplace", action="store_true", help="Overwrite the input file instead of writing a new one.")
	p.add_argument("--dry-run", action="store_true", help="Show what would be removed without writing anything.")
	p.add_argument("--keys", nargs="+", default=None,
		help="Exact top-level keys to remove (overrides automatic 'replay' key detection).")
	p.add_argument("--unsafe-load", action="store_true",
		help="Allow torch.load(weights_only=False) if the safe load fails. Only for files you trust.")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	src: Path = args.path

	if not src.is_file():
		print(f"Error: {src} does not exist or is not a file.", file=sys.stderr)
		return 1
	if args.inplace and args.output is not None:
		print("Error: use either --inplace or --output, not both.", file=sys.stderr)
		return 1

	before = src.stat().st_size
	state = load_artifact(src, args.unsafe_load)
	if not isinstance(state, dict):
		print(f"Error: expected a dict artifact, got {type(state).__name__}.", file=sys.stderr)
		return 1

	print(f"-> Top-level keys: {list(state.keys())}")
	to_remove = find_replay_keys(state, args.keys)

	if not to_remove:
		print("-> No replay buffer entries found. Nothing to do.")
		return 0

	print(f"-> Replay entries to remove: {to_remove}")
	if args.dry_run:
		print("-> Dry run: no file written.")
		return 0

	for k in to_remove:
		del state[k]

	dst = src if args.inplace else (args.output or default_output_path(src))
	dst.parent.mkdir(parents=True, exist_ok=True)

	# Atomic write: never leave a half-written file in place of a good one.
	tmp = dst.with_suffix(dst.suffix + ".tmp")
	try:
		torch.save(state, tmp)
		os.replace(tmp, dst)
	finally:
		if tmp.exists():
			tmp.unlink()

	after = dst.stat().st_size
	print(f"-> Saved: {dst}")
	print(f"-> Size: {human_size(before)} -> {human_size(after)} (saved {human_size(max(before - after, 0))})")
	return 0


if __name__ == "__main__":
	sys.exit(main())