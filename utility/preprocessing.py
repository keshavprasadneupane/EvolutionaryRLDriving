import numpy as np

_OBS_H, _OBS_W = 84, 84  # CarRacing 96x96 after cropping
_sum_scratch: np.ndarray | None = None  # uint16: r+g+b reaches 765, which overflows uint8


def preprocess_observation(
	observation: np.ndarray,
	out: np.ndarray | None = None,
) -> np.ndarray:
	"""
	Crops CarRacing-v3 to 84x84 and converts RGB to grayscale with
	gs = floor((r + g + b) / 3), using integer math only.

	Returns an (84, 84) uint8 frame in [0, 255]. Pass `out` (uint8, (84, 84))
	to write into existing memory. Scaling to [0, 1] happens in ConvNet.forward.
	"""
	global _sum_scratch
	if _sum_scratch is None:
		_sum_scratch = np.empty((_OBS_H, _OBS_W), dtype=np.uint16)
	if out is None:
		out = np.empty((_OBS_H, _OBS_W), dtype=np.uint8)

	cropped = observation[:-12, 6:-6]  # view, no copy
	np.add(cropped[..., 0], cropped[..., 1], out=_sum_scratch, dtype=np.uint16)
	np.add(_sum_scratch, cropped[..., 2], out=_sum_scratch, dtype=np.uint16)
	np.floor_divide(_sum_scratch, 3, out=out, casting="unsafe")  # max 255
	return out


class FrameStack:
	"""
	Circular buffer of the last `n_frames` grayscale frames. Returns a
	(n_frames, H, W) uint8 array, oldest frame first.

	Everything is allocated once, on the first `reset`. The returned array is
	one of two internal buffers that alternate on each call, so `state` and
	`next_state` never alias. A returned array is overwritten by the call after
	next; ReplayBuffer.add copies it, so this is safe in the training loop.
	"""

	def __init__(self, n_frames: int) -> None:
		if n_frames < 1:
			raise ValueError("n_frames must be at least 1.")
		self.n_frames = n_frames
		self._ring: np.ndarray | None = None
		self._out: tuple[np.ndarray, np.ndarray] | None = None
		self._pos = 0   # slot holding the oldest frame (overwritten next)
		self._flip = 0  # which output buffer the next call writes to

	def _allocate(self, frame: np.ndarray) -> None:
		shape = (self.n_frames, *frame.shape)
		self._ring = np.empty(shape, dtype=frame.dtype)
		self._out = (np.empty(shape, dtype=frame.dtype), np.empty(shape, dtype=frame.dtype))

	def reset(self, frame: np.ndarray) -> np.ndarray:
		"""New episode: every slot is filled with the first frame."""
		if (
			self._ring is None
			or self._ring.shape[1:] != frame.shape
			or self._ring.dtype != frame.dtype
		):
			self._allocate(frame)
		assert self._ring is not None
		self._ring[:] = frame
		self._pos = 0
		return self._emit()

	def step(self, frame: np.ndarray) -> np.ndarray:
		"""Overwrites the oldest frame with the newest and returns the stack."""
		if self._ring is None:
			raise RuntimeError("Call reset() before step().")
		self._ring[self._pos] = frame
		self._pos = (self._pos + 1) % self.n_frames
		return self._emit()

	def _emit(self) -> np.ndarray:
		assert self._ring is not None and self._out is not None
		out = self._out[self._flip]
		self._flip ^= 1
		tail = self.n_frames - self._pos
		out[:tail] = self._ring[self._pos:]   # oldest frames
		out[tail:] = self._ring[:self._pos]   # newest frames
		return out