from torch import nn
from network.nnetwork import NNewtork


class FeedForward(NNewtork):
	"""
	A customizable feedforward neural network module.

	Supports dynamic hidden layer sizing, configurable activation functions,
	and custom loss criteria.

	Notes
	-----
	- Defaults to `nn.SmoothL1Loss` if no loss function is provided.
	- Device placement and tensor movement are managed by the base class.
	- Optimization and training loops are intentionally decoupled and managed externally.
	"""
	def __init__(
		self,
		*,
		input_dim: int,
		output_dim: int,
		hidden_dim: list[int] | None = None,
		activation: list[nn.Module] | None = None,
		loss_function: nn.Module | None = None,
		device: str = "cpu",
	) -> None:
		"""Initializes the FeedForward network with a sequential layout.

		Args:
			input_dim: Dimension of the input feature space.
			output_dim: Dimension of the output layer.
			hidden_dim: Sizes of the hidden layers. Defaults to [8].
			activation: Activation modules for hidden layers. Defaults to [nn.ReLU()].
				If fewer activations than hidden layers are provided, the last activation
				is reused.
			loss_function: Loss function module. Defaults to SmoothL1Loss.
			device: Device on which to place the network.
		"""
		if hidden_dim is None:
			hidden_dim = [8]

		if activation is None:
			activation = [nn.ReLU()]

		if hidden_dim and not activation:
			raise ValueError(
				"At least one activation function must be provided when hidden layers exist."
			)

		layers: list[nn.Module] = []
		curr_dim = input_dim
		activation_len = len(activation)

		for i, h_dim in enumerate(hidden_dim):
			layers.append(nn.Linear(curr_dim, h_dim))

			act = activation[i] if i < activation_len else activation[-1]
			layers.append(act)

			curr_dim = h_dim

		layers.append(nn.Linear(curr_dim, output_dim))
		sequential_nn = nn.Sequential(*layers)

		super().__init__(
			model=sequential_nn,
			loss_function=loss_function,
			device=device,
		)
