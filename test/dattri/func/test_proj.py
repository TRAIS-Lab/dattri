"""Unit test for random projection."""

import unittest

import torch
from torch import nn

from dattri.func.projection import (
    ArnoldiProjector,
    BasicProjector,
    ChunkedCudaProjector,
    CudaProjector,
    arnoldi_project,
    random_project,
)

# Check if fast_jl is available
# TODO: Remove this check once we remove fast_jl dependency
try:
    import fast_jl  # noqa: F401
    FAST_JL_AVAILABLE = True
except ImportError:
    FAST_JL_AVAILABLE = False


class TestBasicProjector(unittest.TestCase):
    """Test basic projector functions."""

    def setUp(self):
        """Set up variables for testing."""
        self.feature_dim = 1000
        self.proj_dim = 50
        self.seed = 42
        self.proj_type = "rademacher"
        self.device = "cuda"
        self.projector = None

    def test_basic_projector_shape(self):
        """Test BasicProjector output shape."""
        self.projector = BasicProjector(
            feature_dim=self.feature_dim,
            proj_dim=self.proj_dim,
            seed=self.seed,
            proj_type=self.proj_type,
            device="cpu",
        )

        test_grads = torch.randn(10, self.feature_dim)
        projected_grads = self.projector.project(test_grads, ensemble_id=0)
        assert projected_grads.shape == (10, self.proj_dim)


# TODO: Remove FAST_JL_AVAILABLE check once we remove fast_jl dependency
@unittest.skipUnless(
    torch.cuda.is_available() and FAST_JL_AVAILABLE,
    "CUDA is not available or fast_jl is not installed",
)
class TestCudaProjector(unittest.TestCase):
    """Test cuda projector function."""

    def setUp(self):
        """Set up varibles for testing."""
        self.feature_dim = 100000
        self.proj_dim = 512
        self.seed = 42
        self.proj_type = "rademacher"
        self.device = "cuda:0"
        self.max_batch_size = 32

        self.projector = CudaProjector(
            feature_dim=self.feature_dim,
            proj_dim=self.proj_dim,
            seed=self.seed,
            proj_type=self.proj_type,
            device=self.device,
            max_batch_size=self.max_batch_size,
        )

    def test_project_output_shape(self):
        """Test output shape."""
        grads = torch.randn(64, self.feature_dim, device=self.device)
        grads = torch.randn(64, self.feature_dim, device=self.device)
        ensemble_id = 0
        projected_grads = self.projector.project(grads, ensemble_id)
        assert projected_grads.shape == (64, self.proj_dim)
        assert projected_grads.shape == (64, self.proj_dim)


# TODO: Remove FAST_JL_AVAILABLE check once we remove fast_jl dependency
@unittest.skipUnless(
    torch.cuda.is_available() and FAST_JL_AVAILABLE,
    "CUDA is not available or fast_jl is not installed",
)
class TestChunkedCudaProjector(unittest.TestCase):
    """Test chunked cuda projector function."""

    def setUp(self):
        """Set up varibles for testing."""
        self.device = torch.device("cuda:0")
        self.dtype = torch.float32
        self.proj_dim = 512
        self.max_chunk_size = 5
        self.proj_max_batch_size = 16
        self.feature_dim = 100000
        self.feature_batch_size = 1000
        self.seed = 42
        self.proj_type = "rademacher"
        self.max_batch_size = 32
        self.dim_per_chunk = [5, 5]

        self.projectors = [
            CudaProjector(
                feature_dim=self.feature_dim,
                proj_dim=self.proj_dim,
                seed=self.seed,
                proj_type=self.proj_type,
                device=self.device,
                max_batch_size=self.max_batch_size,
            ),
            CudaProjector(
                feature_dim=self.feature_dim,
                proj_dim=self.proj_dim,
                seed=self.seed,
                proj_type=self.proj_type,
                device=self.device,
                max_batch_size=self.max_batch_size,
            ),
        ]
        self.chunked_projector = ChunkedCudaProjector(
            projector_per_chunk=self.projectors,
            max_chunk_size=self.max_chunk_size,
            dim_per_chunk=self.dim_per_chunk,
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
        }
        ensemble_id = 1
        projected_grads = self.chunked_projector.project(grads, ensemble_id)

        assert projected_grads.shape == (self.feature_batch_size, self.proj_dim)


class TestArnoldiProjector(unittest.TestCase):
    """Test Arnoldi projector functions."""

    def setUp(self):
        """Set up variables for testing."""
        self.feature_dim = 5
        self.vec_dim = 10
        self.proj_dim = 5
        self.device = "cpu"
        self.projector = None

    def test_arnoldi_projector(self):
        """Test ArnoldiProjector functionality and shape."""

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(self.feature_dim)
        # set reg large enough to have some positive eigvals
        reg = 1.0
        self.projector = ArnoldiProjector(
            self.feature_dim,
            self.proj_dim,
            target,
            x,
            regularization=reg,
        )

        vec1 = torch.randn(self.vec_dim, self.feature_dim)
        vec2 = torch.randn(self.vec_dim, self.feature_dim)
        projected_grads1 = self.projector.project(vec1)
        projected_grads2 = self.projector.project(vec2)

        # test the closeness of inner product only
        assert torch.allclose(
            (projected_grads1 @ projected_grads2.T),
            (vec1 @ (torch.diag(1 / (reg - x.sin())))) @ vec2.T,
            rtol=1e-01,
            atol=1e-04,
        )
        # test the shape
        assert projected_grads1.shape == (self.vec_dim, self.feature_dim)
        assert projected_grads2.shape == (self.vec_dim, self.feature_dim)


class SmallModel(nn.Module):
    """A small PyTorch model for testing."""

    def __init__(self):
        """Initialize layers of the model."""
        super(SmallModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

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
        x = x.view(-1, 128 * 4 * 4)
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

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        d_ff: int,
        input_vocab_size: int,
        output_vocab_size: int,
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
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(hidden_size, num_heads, d_ff) for _ in range(num_layers)],
        )
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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model),
        )
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
        x += self.pe[: x.size(0), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """The class for transformer encoder layer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
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
        x += self.dropout(self.norm1(attn_output))
        linear_output = self.linear2(torch.relu(self.linear1(x)))
        return x + self.dropout(self.norm2(linear_output))


class TestGetProjection(unittest.TestCase):
    """Test the random_project function."""

    def setUp(self):
        """Set up varibles for testing."""
        self.small_model = SmallModel()
        self.large_model = LargerModel()
        self.ensemble_id = 0
        self.proj_dim = 512
        self.proj_max_batch_size = 16

    def test_basicprojector(self):
        """Test funcionality of BasicProjetor."""
        test_batch_size = 8
        # mimic gradient
        small_gradient = {}
        for name, p in self.small_model.named_parameters():
            small_gradient[name] = torch.rand(test_batch_size, p.numel())

        # suppose to be BasicProjector
        project = random_project(
            small_gradient,
            test_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device="cpu",
            proj_seed=0,
            use_half_precision=True,
        )

        result_1 = project(small_gradient)
        assert result_1.shape == (test_batch_size, self.proj_dim)

    # TODO: Remove FAST_JL_AVAILABLE check once we remove fast_jl dependency
    @unittest.skipUnless(
        torch.cuda.is_available() and FAST_JL_AVAILABLE,
        "CUDA is not available or fast_jl is not installed",
    )
    def test_cudaprojector(self):
        """Test funcionality of CudaProjetor."""
        test_batch_size = 32
        # mimic gradient
        small_gradient = {}
        for name, p in self.small_model.named_parameters():
            small_gradient[name] = torch.rand(test_batch_size, p.numel())

        # suppose to be CudaProjector
        project = random_project(
            small_gradient,
            test_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device="cuda",
            proj_seed=0,
            use_half_precision=True,
        )

        result_2 = project(small_gradient)
        assert result_2.shape == (test_batch_size, self.proj_dim)

    # TODO: Remove FAST_JL_AVAILABLE check once we remove fast_jl dependency
    @unittest.skipUnless(
        torch.cuda.is_available() and FAST_JL_AVAILABLE,
        "CUDA is not available or fast_jl is not installed",
    )
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

        self.large_transformer = LargeTransformer(
            num_layers,
            hidden_size,
            num_heads,
            d_ff,
            input_vocab_size,
            output_vocab_size,
        )

        # mimic gradient
        large_gradient = {}
        for name, p in self.large_transformer.named_parameters():
            large_gradient[name] = torch.rand(test_batch_size, p.numel())

        # suppose to be ChunkedCudaProjector
        project = random_project(
            large_gradient,
            test_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device="cuda",
            proj_seed=0,
            use_half_precision=True,
        )

        result_3 = project(large_gradient)
        assert result_3.shape == (test_batch_size, self.proj_dim)

    def test_tensor_input_cpu(self):
        """Test the usage of tensor input."""
        test_batch_size = 64

        test_tensor = torch.rand(test_batch_size, 1000)
        # suppose to be BasicProjector
        project = random_project(
            test_tensor,
            test_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device="cpu",
            proj_seed=0,
            use_half_precision=True,
        )

        result = project(test_tensor)
        assert result.shape == (test_batch_size, self.proj_dim)

    # TODO: Remove FAST_JL_AVAILABLE check once we remove fast_jl dependency
    @unittest.skipUnless(
        torch.cuda.is_available() and FAST_JL_AVAILABLE,
        "CUDA is not available or fast_jl is not installed",
    )
    def test_tensor_input_cuda(self):
        """Test the usage of tensor input."""
        test_batch_size = 64

        test_tensor = torch.rand(test_batch_size, 1000)
        # suppose to be CudaProjector
        project = random_project(
            test_tensor,
            test_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device="cuda",
            proj_seed=0,
            use_half_precision=True,
        )

        result = project(test_tensor)
        assert result.shape == (test_batch_size, self.proj_dim)

    # TODO: Remove FAST_JL_AVAILABLE check once we remove fast_jl dependency
    @unittest.skipUnless(
        torch.cuda.is_available() and FAST_JL_AVAILABLE,
        "CUDA is not available or fast_jl is not installed",
    )
    def test_tensor_input_chunked_cuda(self):
        """Test the usage of tensor input."""
        feature_batch_size = 4
        # 0.3B is slighly larger then max_chunk_size (~0.26B)
        test_tensor = torch.rand(feature_batch_size, 300000000)
        # suppose to be ChunkedCudaProjector
        project = random_project(
            test_tensor,
            feature_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device="cuda",
            proj_seed=0,
            use_half_precision=True,
        )

        result = project(test_tensor)
        assert result.shape == (feature_batch_size, self.proj_dim)

    def test_arnoldi_project(self):
        """Test the funcitonality of arnoldi_project."""
        feature_dim = 10
        proj_dim = 5
        vec_dim = 20

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(feature_dim)
        vec = torch.randn(vec_dim, feature_dim)

        projector = arnoldi_project(
            feature_dim,
            target,
            x,
            proj_dim=proj_dim,
        )

        result = projector(vec)

        assert result.shape == (vec_dim, proj_dim)


if __name__ == "__main__":
    unittest.main()
