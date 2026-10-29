import torch
from torch import nn
from network.nnetwork import NNewtork


class ConvNet(NNewtork):
	"""
	A convolutional neural network specifically tailored for the CarRacing environment.

	Notes
	-----
	- Designed for 84x84 inputs (cropped to remove dashboard/text). `in_channels`
	  is the number of stacked grayscale frames (N), not RGB channels.
	- Expects raw pixel values in [0, 255] (uint8). The division by 255 happens
	  inside forward(), so replay buffers can store frames as uint8. Do not
	  pre-normalize inputs, or they will be scaled twice.
	- Device placement and tensor movement are managed by the base class.
	"""
	def __init__(
		self,
		*,
		in_channels: int = 4,
		output_dim: int = 5,
		hidden_dim: int = 224,
		loss_function: nn.Module | None = None,
		device: str = "cpu",
	) -> None:
		# 1. Build local components first (before super().__init__())
		conv = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=4, stride=2),
			nn.ReLU(),
		)
		n_features = 32 * 9 * 9

		fc = nn.Sequential(
			nn.Linear(n_features, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
		)

		sequential_nn = nn.Sequential(
			conv,
			nn.Flatten(),
			fc,
		)

		super().__init__(
			model=sequential_nn,
			loss_function=loss_function,
			device=device,
		)

		self.conv = conv
		self.fc = fc
		self._n_features = n_features

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Scales pixels from [0, 255] to [0, 1], then runs the network."""
		return self.nn(x / 255.0)