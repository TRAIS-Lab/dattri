"""Unit test for random projection."""

import unittest

import torch
from torch import nn

from dattri.utils import vectorize
from dattri.func.random_projection import (
    BasicProjector,
    ChunkedCudaProjector,
    CudaProjector,
    get_projection
    get_projection,
)
from dattri.func.utils import _vectorize as vectorize  # noqa: PLC2701


class TestVectorize(unittest.TestCase):
    """Test vectorize function."""

    def setUp(self):
        """Set up variables for testing."""
        self.g_dict_1 = {
            "a": torch.randn(10, 5),
            "b": torch.randn(10, 3),
        }

        self.g_dict_2 = {
            "a": torch.tensor([[1., 2., 3.], [6., 7., 8.]]),
            "b": torch.tensor([[4., 5.], [9., 10.]]),
        }

        self.expected_shape = (10, 8)

    def test_vectorize_shape_without_arr(self):
        """Test the shape without arr argument."""
        result = vectorize(self.g_dict_1, device="cpu")
        assert result.shape == self.expected_shape

    def test_vectorize_shape_with_arr(self):
        """Test the shape with arr argument."""
        arr = torch.empty(size=self.expected_shape)
        result = vectorize(self.g_dict_1, arr=arr, device="cpu")
        assert result is arr
        assert result.shape == self.expected_shape

    def test_vectorize_value(self):
        """Test the value correctness."""
        result = vectorize(self.g_dict_2, device="cpu")
        answer = torch.tensor([[1., 2., 3., 4., 5.],
                               [6., 7., 8., 9., 10.]])
        assert (torch.equal(result, answer))


class TestBasicProjector(unittest.TestCase):
    """Test basic projector functions."""

    def setUp(self):
        """Set up variables for testing."""
        self.grad_dim = 1000
        self.proj_dim = 50
        self.seed = 42
        self.proj_type = "rademacher"
        self.device = "cuda"
        self.projector = None

    def test_basic_projector_shape(self):
        """Test BasicProjector output shape."""
        self.projector = BasicProjector(
            grad_dim=self.grad_dim,
            proj_dim=self.proj_dim,
            seed=self.seed,
            proj_type=self.proj_type,
            device="cpu",
        )

        test_grads = torch.randn(10, self.grad_dim)
        projected_grads = self.projector.project(test_grads, model_id=0)
        assert projected_grads.shape == (10, self.proj_dim)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class TestCudaProjector(unittest.TestCase):
    """Test cuda projector function."""
    def setUp(self):
        """Set up varibles for testing."""
        self.grad_dim = 100000
        self.proj_dim = 512
        self.seed = 42
        self.proj_type = "rademacher"
        self.device = "cuda:0"
        self.max_batch_size = 32

        self.projector = CudaProjector(
                grad_dim=self.grad_dim,
                proj_dim=self.proj_dim,
                seed=self.seed,
                proj_type=self.proj_type,
                device=self.device,
                max_batch_size=self.max_batch_size,
            )

    def test_project_output_shape(self):
        """Test output shape."""
        grads = torch.randn(64, self.grad_dim, device=self.device)
        grads = torch.randn(64, self.grad_dim, device=self.device)
        model_id = 0
        projected_grads = self.projector.project(grads, model_id)
        assert projected_grads.shape == (64, self.proj_dim)
        assert projected_grads.shape == (64, self.proj_dim)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class TestChunkedCudaProjector(unittest.TestCase):
    """Test chunked cuda projector function."""
    def setUp(self):
        """Set up varibles for testing."""
        self.device = torch.device("cuda:0")
        self.dtype = torch.float32
        self.proj_dim = 512
        self.max_chunk_size = 5
        self.proj_max_batch_size = 4
        self.grad_dim = 100000
        self.feature_batch_size = 1000
        self.feature_batch_size = 1000
        self.seed = 42
        self.proj_type = "rademacher"
        self.max_batch_size = 32
        self.params_per_chunk = [5, 5]

        self.projectors = [
            CudaProjector(grad_dim=self.grad_dim,
                          proj_dim=self.proj_dim,
                          seed=self.seed,
                          proj_type=self.proj_type,
                          device=self.device,
                          max_batch_size=self.max_batch_size),
            CudaProjector(grad_dim=self.grad_dim,
                          proj_dim=self.proj_dim,
                          seed=self.seed,
                          proj_type=self.proj_type,
                          device=self.device,
                          max_batch_size=self.max_batch_size),
        ]
        self.chunked_projector = ChunkedCudaProjector(
            projector_per_chunk=self.projectors,
            max_chunk_size=self.max_chunk_size,
            params_per_chunk=self.params_per_chunk,
            feature_batch_size=self.feature_batch_size,
            feature_batch_size=self.feature_batch_size,
            proj_max_batch_size=self.proj_max_batch_size,
            device=self.device,
            dtype=self.dtype,
        )

    def test_project_output_shape(self):
        """Test the projection output shape."""
        grads = {
            "grad1": torch.randn(self.feature_batch_size, 3, device=self.device),
            "grad2": torch.randn(self.feature_batch_size, 2, device=self.device),
            "grad3": torch.randn(self.feature_batch_size, 3, device=self.device),
            "grad4": torch.randn(self.feature_batch_size, 2, device=self.device),
            "grad1": torch.randn(self.feature_batch_size, 3, device=self.device),
            "grad2": torch.randn(self.feature_batch_size, 2, device=self.device),
            "grad3": torch.randn(self.feature_batch_size, 3, device=self.device),
            "grad4": torch.randn(self.feature_batch_size, 2, device=self.device),
        }
        model_id = 1
        projected_grads = self.chunked_projector.project(grads, model_id)

        assert projected_grads.shape == (self.feature_batch_size, self.proj_dim)


class SmallModel(nn.Module):
    """A small PyTorch model for testing."""
    def __init__(self):
        """Initialize layers of the model."""
        super(SmallModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor):
        """Model's forward pass.

        Args:
            x (torch.Tensor): model input.

        Returns:
            torch.Tensor: model output.
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class LargerModel(nn.Module):
    """A large PyTorch model for testing."""
    def __init__(self):
        """Initialize layers of the model."""
        super(LargerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        """Model's forward pass.

        Args:
            x (torch.Tensor): model input.

        Returns:
            torch.Tensor: model output.
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 512 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class LargeTransformer(nn.Module):
    """A large transformer for testing."""
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int,
                 d_ff: int, input_vocab_size: int, output_vocab_size: int,
        ) -> None:
        """Initialize the transformer model.

        Args:
            num_layers (int): Number of transformer layers.
            hidden_size (int): Hidden size of the transformer.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feedforward network.
            input_vocab_size (int): Size of the input vocabulary.
            output_vocab_size (int): Size of the output vocabulary.
        """
        super(LargeTransformer, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, x):
        """Model's forward pass.

        Args:
            x (torch.Tensor): model input.

        Returns:
            torch.Tensor: model output.
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = [layer(x) for layer in self.encoder_layers]
        return self.fc(x)

class PositionalEncoding(nn.Module):
    """The class for positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize the positional encoding.

        Args:
            d_model (int): Dimensionality of the model.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
                              (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Model's forward pass.

        Args:
            x (torch.Tensor): model input.

        Returns:
            torch.Tensor: model output.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    """The class for transformer encoder layer."""
    def __init__(self, hidden_size: int, num_heads: int,
                 d_ff: int, dropout: float = 0.1,
        ) -> None:
        """Initialize the encoder layer.

        Args:
            hidden_size (int): Dimensionality of the input and output features.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feedforward layer.
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, d_ff)
        self.linear2 = nn.Linear(d_ff, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Model's forward pass.

        Args:
            x (torch.Tensor): model input.

        Returns:
            torch.Tensor: model output.
        """
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(self.norm1(attn_output))
        linear_output = self.linear2(torch.relu(self.linear1(x)))
        return x + self.dropout(self.norm2(linear_output))

class TestGetProjection(unittest.TestCase):
    """Test the get_projection function."""
    def setUp(self):
        """Set up varibles for testing."""
        self.small_model = SmallModel()
        self.large_model = LargerModel()
        self.model_id = 0
        self.proj_dim = 1024
        self.proj_max_batch_size = 32


    def test_basicprojector(self):
        """Test funcionality of BasicProjetor."""
        test_batch_size = 8
        # mimic gradient
        small_gradient = {}
        for name, p in self.small_model.named_parameters():
            small_gradient[name] = torch.rand(test_batch_size, p.numel())

        # suppose to be BasicProjector
        result_1 = get_projection(self.small_model, self.model_id, small_gradient,
                                  test_batch_size, "cpu",
                                  self.proj_dim, self.proj_max_batch_size,
                                  proj_seed=0 , use_half_precision=True)

        assert result_1.shape == (test_batch_size, self.proj_dim)


    def test_cudaprojector(self):
        """Test funcionality of CudaProjetor."""
        test_batch_size = 32
        # mimic gradient
        small_gradient = {}
        for name, p in self.small_model.named_parameters():
            small_gradient[name] = torch.rand(test_batch_size, p.numel()).cuda()

        # suppose to be CudaProjector
        result_2 = get_projection(self.small_model, self.model_id,
                                  small_gradient, test_batch_size, "cuda",
                                  self.proj_dim, self.proj_max_batch_size,
                                  proj_seed=0 , use_half_precision=True)

        assert result_2.shape == (test_batch_size, self.proj_dim)


    def test_chunkedcudaprojector(self):
        """Test funcionality of ChunkedCudaProjector."""
        test_batch_size = 64
        # Define parameters
        num_layers = 16
        hidden_size = 1024
        num_heads = 16
        d_ff = 2048
        input_vocab_size = 1000
        output_vocab_size = 1000

        self.large_transformer = LargeTransformer(num_layers, hidden_size,
                                                  num_heads, d_ff, input_vocab_size,
                                                  output_vocab_size)

        # mimic gradient
        large_gradient = {}
        for name, p in self.large_transformer.named_parameters():
            large_gradient[name] = torch.rand(test_batch_size, p.numel())

        # suppose to be ChunkedCudaProjector
        result_3 = get_projection(self.large_transformer, self.model_id,
                                  large_gradient, test_batch_size, "cuda",
                                  self.proj_dim, self.proj_max_batch_size,
                                  proj_seed=0 , use_half_precision=True)

        assert result_3.shape == (test_batch_size, self.proj_dim)


if __name__ == "__main__":
    unittest.main()
