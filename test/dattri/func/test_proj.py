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


def compute_pairwise_distance_metrics(batch_vec, batch_vec_p):
    """Compute relative error between pairwise distances.

    Validates that random projections preserve pairwise distances,
    which is a key property of Johnson-Lindenstrauss transforms and similar
    random projection methods.

    Arguments:
        batch_vec (Tensor): original data (batch of vectors)
        batch_vec_p (Tensor): projected data (batch of projected vectors)

    Returns:
        relative_error (float): average relative error between original and
            projected pairwise distances
    """
    # Convert to float32 to avoid numerical overflow with float16
    # float16 has max value ~65504, and pairwise distances can easily overflow
    if batch_vec.dtype in (torch.float16, torch.bfloat16):
        batch_vec = batch_vec.float()
        batch_vec_p = batch_vec_p.float()

    # Compute pairwise distances
    original_distances = torch.cdist(batch_vec, batch_vec, p=2)
    projected_distances = torch.cdist(batch_vec_p, batch_vec_p, p=2)

    # Avoid division by zero for any zero distances in the original data
    zero_threshold = 1e-8
    mask = original_distances > zero_threshold

    # Compute Relative Error
    relative_errors = torch.abs(
        (original_distances[mask] - projected_distances[mask])
        / original_distances[mask],
    )
    return torch.mean(relative_errors).item()


class TestBasicProjector(unittest.TestCase):
    """Test BasicProjector class for CPU-based random projections.

    BasicProjector is a simple block-wise implementation used for CPU devices.
    It generates projection matrices on-device in blocks and accumulates results.
    This class tests the basic functionality and output shape correctness.
    """

    def setUp(self):
        """Set up test parameters for BasicProjector."""
        self.feature_dim = 1000
        self.proj_dim = 50
        self.seed = 42
        self.proj_type = "rademacher"
        self.device = torch.device("cpu")
        self.projector = None

    def test_basic_projector_shape(self):
        """Test that BasicProjector produces correct output shape.

        Verifies that projecting a batch of 10 features from feature_dim to proj_dim
        produces the expected output shape of (batch_size, proj_dim).
        """
        self.projector = BasicProjector(
            feature_dim=self.feature_dim,
            proj_dim=self.proj_dim,
            seed=self.seed,
            proj_type=self.proj_type,
            device=self.device,
        )

        test_grads = torch.randn(10, self.feature_dim)
        projected_grads = self.projector.project(test_grads, ensemble_id=0)
        expected = (10, self.proj_dim)
        assert projected_grads.shape == expected, (
            f"Shape mismatch: expected {expected}, got {projected_grads.shape}"
        )


@unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA is not available",
)
class TestCudaProjector(unittest.TestCase):
    """Test CudaProjector class for GPU-accelerated random projections.

    CudaProjector is a performant implementation optimized for CUDA devices
    with compute capability >= 7.0. It supports three projection types:
    - sjlt: Sparse Johnson-Lindenstrauss Transform (most efficient)
    - normal: Gaussian random projection
    - rademacher: Rademacher random projection

    This class tests the basic functionality with the sjlt projection type.
    """

    def setUp(self):
        """Set up test parameters for CudaProjector with sjlt projection."""
        self.feature_dim = 100000
        self.proj_dim = 512
        self.seed = 42
        self.proj_type = "sjlt"
        self.device = torch.device("cuda")
        self.max_batch_size = 32

        self.projector = CudaProjector(
            feature_dim=self.feature_dim,
            proj_dim=self.proj_dim,
            seed=self.seed,
            proj_type=self.proj_type,
            device=self.device,
            max_batch_size=self.max_batch_size,
            dtype=torch.float32,
        )

    def test_project_output_shape(self):
        """Test that CudaProjector produces correct output shape.

        Verifies that projecting a batch of 64 high-dimensional features
        (100k dimensions) produces the expected output shape of (64, proj_dim).
        """
        grads = torch.randn(64, self.feature_dim, device=self.device)
        ensemble_id = 0
        projected_grads = self.projector.project(grads, ensemble_id)
        expected = (64, self.proj_dim)
        assert projected_grads.shape == expected, (
            f"Shape mismatch: expected {expected}, got {projected_grads.shape}"
        )


@unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA is not available",
)
class TestChunkedCudaProjector(unittest.TestCase):
    """Test ChunkedCudaProjector for memory-efficient large-scale projections.

    ChunkedCudaProjector is used when the feature dimension multiplied by batch size
    is too large to fit in GPU memory. It splits features into chunks, projects each
    chunk separately using individual CudaProjectors, and accumulates the results.

    This is particularly useful for very large models where the total number of
    parameters (gradient dimensions) exceeds memory capacity when combined with
    the required batch size.
    """

    def setUp(self):
        """Set up test parameters for ChunkedCudaProjector with multiple chunks.

        Creates a chunked projector that splits 100k features into two 50k chunks,
        each with its own CudaProjector instance.
        """
        self.device = torch.device("cuda")
        self.dtype = torch.float32
        self.proj_dim = 512
        self.max_chunk_size = 50000
        self.proj_max_batch_size = 16
        self.feature_dim = 100000
        self.feature_batch_size = 1000
        self.seed = 42
        self.proj_type = "sjlt"
        self.max_batch_size = 32
        self.dim_per_chunk = [self.max_chunk_size, self.max_chunk_size]

        self.projectors = [
            CudaProjector(
                feature_dim=self.max_chunk_size,
                proj_dim=self.proj_dim,
                seed=self.seed,
                proj_type=self.proj_type,
                device=self.device,
                max_batch_size=self.max_batch_size,
            ),
            CudaProjector(
                feature_dim=self.max_chunk_size,
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
        """Test that ChunkedCudaProjector produces correct output shape with dict input.

        Creates a dictionary of gradient tensors that span multiple chunks
        (30k + 20k + 20k + 30k = 100k total dimensions) and verifies the
        projected output has the correct shape (feature_batch_size, proj_dim).
        """
        grads = {
            "grad1": torch.randn(self.feature_batch_size, 30000, device=self.device),
            "grad2": torch.randn(self.feature_batch_size, 20000, device=self.device),
            "grad3": torch.randn(self.feature_batch_size, 20000, device=self.device),
            "grad4": torch.randn(self.feature_batch_size, 30000, device=self.device),
        }
        ensemble_id = 1
        projected_grads = self.chunked_projector.project(grads, ensemble_id)

        expected_shape = (self.feature_batch_size, self.proj_dim)
        assert projected_grads.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {projected_grads.shape}"
        )


class TestArnoldiProjector(unittest.TestCase):
    """Test ArnoldiProjector for inverse Hessian-based projections.

    ArnoldiProjector uses Arnoldi iteration to compute an approximate eigenspace
    decomposition of the inverse Hessian matrix. This provides a more informed
    projection that captures the geometry of the loss landscape, unlike random
    projections which are geometry-agnostic.

    The projection is based on the top-k eigenvalues and eigenvectors of the
    inverse Hessian, making it particularly useful for influence function
    calculations and other second-order optimization tasks.
    """

    def setUp(self):
        """Set up test parameters for ArnoldiProjector."""
        self.feature_dim = 5
        self.vec_dim = 10
        self.proj_dim = 5
        self.device = torch.device("cpu")
        self.projector = None

    def test_arnoldi_projector(self):
        """Test ArnoldiProjector functionality and output shape.

        Tests that:
        1. The projector correctly approximates the inverse Hessian projection
           by verifying inner products match the expected regularized Hessian
        2. Output shape is correct (vec_dim, feature_dim)

        Uses a simple target function f(x) = sum(sin(x)) with regularization
        to ensure positive eigenvalues.
        """

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
        expected_shape = (self.vec_dim, self.feature_dim)
        assert projected_grads1.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {projected_grads1.shape}"
        )
        assert projected_grads2.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {projected_grads2.shape}"
        )


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


class TestRandomProjectionFactory(unittest.TestCase):
    """Test the random_project factory function and automatic projector selection.

    The random_project function is a high-level factory that automatically selects
    the appropriate projector type based on:
    - Device (CPU -> BasicProjector, CUDA -> CudaProjector/ChunkedCudaProjector)
    - Feature dimensions and batch size (determines if chunking is needed)
    - Input type (dictionary of gradients vs. single tensor)

    This test class verifies that the factory function correctly instantiates
    projectors and handles different model sizes and input formats.
    """

    def setUp(self):
        """Set up test models and parameters for random_project tests."""
        self.small_model = SmallModel()
        self.large_model = LargerModel()
        self.ensemble_id = 0
        self.proj_dim = 512
        self.proj_max_batch_size = 16

    def test_basicprojector(self):
        """Test random_project creates BasicProjector for small CPU workloads.

        Verifies that random_project correctly instantiates a BasicProjector
        when using CPU device with a small model, and produces correct output
        shape and dtype.
        """
        test_batch_size = 8
        # mimic gradient
        small_gradient = {}
        for name, p in self.small_model.named_parameters():
            small_gradient[name] = torch.rand(
                test_batch_size,
                p.numel(),
                dtype=torch.float16,
            )

        # suppose to be BasicProjector
        project = random_project(
            small_gradient,
            test_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=torch.device("cpu"),
            proj_type="normal",
            proj_seed=0,
        )

        result_1 = project(small_gradient)
        expected_shape = (test_batch_size, self.proj_dim)
        assert result_1.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {result_1.shape}"
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_cudaprojector(self):
        """Test random_project creates CudaProjector for CUDA workloads.

        Verifies that random_project correctly instantiates a CudaProjector
        when using CUDA device with a small model, and produces correct output
        shape and dtype. Uses a larger batch size (32) suitable for GPU.
        """
        test_batch_size = 32
        # mimic gradient
        small_gradient = {}
        for name, p in self.small_model.named_parameters():
            small_gradient[name] = torch.rand(
                test_batch_size,
                p.numel(),
                dtype=torch.float16,
            )

        # suppose to be CudaProjector
        project = random_project(
            small_gradient,
            test_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=torch.device("cuda"),
            proj_seed=0,
        )

        result_2 = project(small_gradient)
        expected_shape = (test_batch_size, self.proj_dim)
        assert result_2.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {result_2.shape}"
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_chunkedcudaprojector(self):
        """Test random_project creates ChunkedCudaProjector for very large models.

        Verifies that random_project automatically selects ChunkedCudaProjector
        when the model is large enough (large transformer with ~200M parameters)
        that chunking is required for memory efficiency. Tests with a batch size
        of 64 on a model with many parameters.
        """
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
            large_gradient[name] = torch.rand(
                test_batch_size,
                p.numel(),
                dtype=torch.float16,
            )

        # suppose to be ChunkedCudaProjector
        project = random_project(
            large_gradient,
            test_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=torch.device("cuda"),
            proj_type="sjlt",
            proj_seed=0,
        )

        result_3 = project(large_gradient)
        expected_shape = (test_batch_size, self.proj_dim)
        assert result_3.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {result_3.shape}"
        )

    def test_tensor_input_cpu(self):
        """Test random_project with direct tensor input on CPU.

        Verifies that random_project correctly handles tensor input (not dict)
        and creates a BasicProjector for CPU. This is useful when projecting
        pre-flattened feature vectors rather than model gradients.
        """
        test_batch_size = 64

        test_tensor = torch.rand(test_batch_size, 1000, dtype=torch.float16)
        # suppose to be BasicProjector
        project = random_project(
            test_tensor,
            test_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=torch.device("cpu"),
            proj_type="normal",
            proj_seed=0,
        )

        result = project(test_tensor)
        expected_shape = (test_batch_size, self.proj_dim)
        assert result.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {result.shape}"
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_tensor_input_cuda(self):
        """Test random_project with direct tensor input on CUDA.

        Verifies that random_project correctly handles tensor input on GPU
        and creates a CudaProjector. Tests with moderate-sized tensors.
        """
        test_batch_size = 64

        test_tensor = torch.rand(test_batch_size, 1000, dtype=torch.float16)
        # suppose to be CudaProjector
        project = random_project(
            test_tensor,
            test_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=torch.device("cuda"),
            proj_seed=0,
        )

        result = project(test_tensor)
        expected_shape = (test_batch_size, self.proj_dim)
        assert result.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {result.shape}"
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_tensor_input_chunked_cuda(self):
        """Test random_project with very large tensor input requiring chunking.

        Verifies that random_project automatically selects ChunkedCudaProjector
        when the input tensor is extremely large (300M dimensions). This tests
        the automatic chunking logic for memory-constrained scenarios.
        """
        feature_batch_size = 4
        # 0.3B is slightly larger then max_chunk_size (~0.26B)
        test_tensor = torch.rand(feature_batch_size, 300000000, dtype=torch.float16)
        # suppose to be ChunkedCudaProjector
        project = random_project(
            test_tensor,
            feature_batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=torch.device("cuda"),
            proj_type="sjlt",
            proj_seed=0,
        )

        result = project(test_tensor)
        expected_shape = (feature_batch_size, self.proj_dim)
        assert result.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {result.shape}"
        )

    def test_arnoldi_project(self):
        """Test the arnoldi_project factory function.

        Verifies that the arnoldi_project factory function correctly creates
        an ArnoldiProjector and returns a callable projection function that
        produces the expected output shape (vec_dim, proj_dim).
        """
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

        expected_shape = (vec_dim, proj_dim)
        assert result.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {result.shape}"
        )


class TestProjectionCombinationsCPU(unittest.TestCase):
    """Test CPU random projections with all proj_type and dtype combinations.

    Test matrix (proj_type x dtype):
        - [normal, rademacher, random_mask] x
          [float64, float32, float16, bfloat16].
    """

    def setUp(self):
        """Set up variables for testing."""
        self.small_model = SmallModel()
        self.proj_dim = 512
        self.proj_max_batch_size = 16
        self.test_batch_size = 8
        self.feature_dim = 1000

    def _test_projection_combination(
        self,
        device_str,
        proj_type,
        dtype,
        batch_size=None,
    ):
        """Helper method to test a specific combination of device, proj_type, and dtype.

        Args:
            device_str (str): 'cpu' or 'cuda'
            proj_type (str): 'normal', 'rademacher', or 'sjlt'
            dtype (torch.dtype): Data type to test
            batch_size (int): Batch size to use, defaults to self.test_batch_size
        """
        if batch_size is None:
            batch_size = self.test_batch_size

        device = torch.device(device_str)

        # Test with dictionary input (mimicking gradients)
        gradient_dict = {}
        for name, p in self.small_model.named_parameters():
            gradient_dict[name] = torch.rand(
                batch_size,
                p.numel(),
                dtype=dtype,
            )

        project = random_project(
            gradient_dict,
            batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=device,
            proj_seed=0,
            proj_type=proj_type,
        )

        result = project(gradient_dict)
        expected_shape = (batch_size, self.proj_dim)
        assert result.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {result.shape}"
        )
        assert result.dtype == dtype, (
            f"Dtype mismatch: expected {dtype}, got {result.dtype}"
        )

        # Test with tensor input
        test_tensor = torch.rand(batch_size, self.feature_dim, dtype=dtype)
        project_tensor = random_project(
            test_tensor,
            batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=device,
            proj_seed=0,
            proj_type=proj_type,
        )

        result_tensor = project_tensor(test_tensor)
        expected_tensor_shape = (batch_size, self.proj_dim)
        assert result_tensor.shape == expected_tensor_shape, (
            f"Shape mismatch: expected {expected_tensor_shape}, "
            f"got {result_tensor.shape}"
        )
        assert result_tensor.dtype == dtype, (
            f"Dtype mismatch: expected {dtype}, got {result_tensor.dtype}"
        )

        # Validate pairwise distance preservation
        # Use a smaller batch for distance computation to avoid memory issues
        distance_batch_size = min(batch_size, 32)
        distance_test_tensor = torch.rand(
            distance_batch_size,
            self.feature_dim,
            dtype=dtype,
        )
        if device_str == "cuda":
            distance_test_tensor = distance_test_tensor.to(device)

        projected_result = project_tensor(distance_test_tensor)

        # Compute pairwise distance preservation metric
        relative_error = compute_pairwise_distance_metrics(
            distance_test_tensor.cpu(),
            projected_result.cpu(),
        )

        # Expected relative error bounds based on projection type
        max_relative_error = 0.5 if proj_type == "random_mask" else 0.05

        assert relative_error < max_relative_error, (
            f"Pairwise distance check failed: relative error ={relative_error:.4f} "
            f"exceeds threshold={max_relative_error} for {proj_type} with {dtype}"
        )

    # ==================== CPU Tests ====================
    # CPU supports: normal, rademacher, random_mask
    # Each tested with: float64, float32, float16, bfloat16

    def test_cpu_projections(self):
        """Test CPU BasicProjector with all projection types and dtypes."""
        proj_types = ["normal", "rademacher", "random_mask"]
        dtypes = [torch.float64, torch.float32, torch.float16, torch.bfloat16]

        for proj_type in proj_types:
            for dtype in dtypes:
                with self.subTest(proj_type=proj_type, dtype=dtype):
                    self._test_projection_combination("cpu", proj_type, dtype)


@unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA is not available",
)
class TestProjectionCombinationsCUDA(unittest.TestCase):
    """Test CUDA random projections with all proj_type and dtype combinations.

    Test matrix (proj_type x dtype):
        - [sjlt, normal, rademacher, random_mask, grass] x
          [float64, float32, float16, bfloat16].
    """

    def setUp(self):
        """Set up variables for testing."""
        self.small_model = SmallModel()
        self.proj_dim = 512
        self.proj_max_batch_size = 16
        self.test_batch_size = 8
        self.feature_dim = 1000

    def _test_projection_combination(
        self,
        device_str,
        proj_type,
        dtype,
        batch_size=None,
    ):
        """Helper method to test a specific combination of device, proj_type, and dtype.

        Args:
            device_str (str): 'cpu' or 'cuda'
            proj_type (str): 'normal', 'rademacher', or 'sjlt'
            dtype (torch.dtype): Data type to test
            batch_size (int): Batch size to use, defaults to self.test_batch_size
        """
        if batch_size is None:
            batch_size = self.test_batch_size

        device = torch.device(device_str)

        # Test with dictionary input (mimicking gradients)
        gradient_dict = {}
        for name, p in self.small_model.named_parameters():
            gradient_dict[name] = torch.rand(
                batch_size,
                p.numel(),
                dtype=dtype,
            )

        project = random_project(
            gradient_dict,
            batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=device,
            proj_seed=0,
            proj_type=proj_type,
        )

        result = project(gradient_dict)
        expected_shape = (batch_size, self.proj_dim)
        assert result.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {result.shape}"
        )
        assert result.dtype == dtype, (
            f"Dtype mismatch: expected {dtype}, got {result.dtype}"
        )

        # Test with tensor input
        test_tensor = torch.rand(batch_size, self.feature_dim, dtype=dtype)
        project_tensor = random_project(
            test_tensor,
            batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=device,
            proj_seed=0,
            proj_type=proj_type,
        )

        result_tensor = project_tensor(test_tensor)
        expected_tensor_shape = (batch_size, self.proj_dim)
        assert result_tensor.shape == expected_tensor_shape, (
            f"Shape mismatch: expected {expected_tensor_shape}, "
            f"got {result_tensor.shape}"
        )
        assert result_tensor.dtype == dtype, (
            f"Dtype mismatch: expected {dtype}, got {result_tensor.dtype}"
        )

        # Validate pairwise distance preservation
        # Use a smaller batch for distance computation to avoid memory issues
        distance_batch_size = min(batch_size, 32)
        distance_test_tensor = torch.rand(
            distance_batch_size,
            self.feature_dim,
            dtype=dtype,
        )
        if device_str == "cuda":
            distance_test_tensor = distance_test_tensor.to(device)

        projected_result = project_tensor(distance_test_tensor)

        # Compute pairwise distance preservation metric
        relative_error = compute_pairwise_distance_metrics(
            distance_test_tensor.cpu(),
            projected_result.cpu(),
        )

        # Expected relative error bounds based on projection type
        max_relative_error = 0.5 if proj_type == "random_mask" else 0.05

        assert relative_error < max_relative_error, (
            f"Pairwise distance check failed: relative error ={relative_error:.4f} "
            f"exceeds threshold={max_relative_error} for {proj_type} with {dtype}"
        )

    def test_cuda_projections(self):
        """Test CUDA CudaProjector with all projection types and dtypes."""
        proj_types = ["sjlt", "normal", "rademacher", "random_mask", "grass"]
        dtypes = [torch.float64, torch.float32, torch.float16, torch.bfloat16]

        for proj_type in proj_types:
            for dtype in dtypes:
                with self.subTest(proj_type=proj_type, dtype=dtype):
                    self._test_projection_combination(
                        "cuda",
                        proj_type,
                        dtype,
                        batch_size=32,
                    )

    def test_cuda_grass_with_multiplier(self):
        """Test CUDA CudaProjector with GraSS projection using custom multipliers."""
        device = torch.device("cuda")
        batch_size = 16

        # Test grass_2
        gradient_dict = {}
        for name, p in self.small_model.named_parameters():
            gradient_dict[name] = torch.rand(
                batch_size,
                p.numel(),
                dtype=torch.float32,
            )

        project = random_project(
            gradient_dict,
            batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=device,
            proj_seed=0,
            proj_type="grass_2",
        )

        result = project(gradient_dict)
        expected_shape = (batch_size, self.proj_dim)
        assert result.shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {result.shape}"
        )

        # Test grass_8
        project_8 = random_project(
            gradient_dict,
            batch_size,
            self.proj_dim,
            self.proj_max_batch_size,
            device=device,
            proj_seed=0,
            proj_type="grass_8",
        )

        result_8 = project_8(gradient_dict)
        expected_shape_8 = (batch_size, self.proj_dim)
        assert result_8.shape == expected_shape_8, (
            f"Shape mismatch: expected {expected_shape_8}, got {result_8.shape}"
        )

        # Different multipliers should produce different results
        assert not torch.allclose(result, result_8), (
            "Different multipliers should produce different results"
        )


if __name__ == "__main__":
    unittest.main()
