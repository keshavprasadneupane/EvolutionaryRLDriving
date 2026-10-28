import torch
from torch import nn
from network.nnetwork import NNewtork


class ConvNet(NNewtork):
	"""
	A convolutional neural network specifically tailored for the CarRacing environment.

	Notes
	-----
	- Designed specifically for 84x84 image inputs (cropped to remove dashboard/text)
	  downsampled to 7x7x64.
	- Expects raw pixel values in [0, 255] (uint8 frames cast to float). The
	  division by 255 happens inside forward(), so replay buffers can store
	  frames as uint8. Do not pre-normalize inputs, or they will be scaled twice.
	- Device placement and tensor movement are managed by the base class.
	"""
	def __init__(
		self,
		*,
		in_channels: int = 3,
		output_dim: int = 5,
		hidden_dim: int = 256,
		loss_function: nn.Module | None = None,
		device: str = "cpu",
	) -> None:
		# sequential_nn = nn.Sequential(
		# 	nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
		# 	nn.ReLU(),
		# 	nn.Conv2d(32, 64, kernel_size=4, stride=2),
		# 	nn.ReLU(),
		# 	nn.Conv2d(64, 64, kernel_size=3, stride=1),
		# 	nn.ReLU(),
		# 	nn.Flatten(),
		# 	nn.Linear(7 * 7 * 64, hidden_dim),
		# 	nn.ReLU(),
		# 	nn.Linear(hidden_dim, output_dim),
		# )
		sequential_nn = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(9 * 9 * 32, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
		)
		super().__init__(
			model=sequential_nn,
			loss_function=loss_function,
			device=device,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Scales pixels from [0, 255] to [0, 1], then runs the network."""
		return self.nn(x / 255.0)