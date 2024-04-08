"""Unit test for random projection."""
import sys
import unittest

import torch

sys.path.append("/u/tli3/dattri_test/dattri")
from dattri.func.random_projection import (
    BasicProjector,
    ChunkedCudaProjector,
    CudaProjector,
    vectorize,
)


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
        grads = torch.randn(16, self.grad_dim, device=self.device)
        model_id = 0
        projected_grads = self.projector.project(grads, model_id)
        assert projected_grads.shape == (16, self.proj_dim)

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
            proj_max_batch_size=self.proj_max_batch_size,
            device=self.device,
            dtype=self.dtype,
        )

    def test_project_output_shape(self):
        """Test the projection output shape."""
        grads = {
            "grad1": torch.randn(self.proj_max_batch_size, 3, device=self.device),
            "grad2": torch.randn(self.proj_max_batch_size, 2, device=self.device),
            "grad3": torch.randn(self.proj_max_batch_size, 3, device=self.device),
            "grad4": torch.randn(self.proj_max_batch_size, 2, device=self.device),
        }
        model_id = 1
        projected_grads = self.chunked_projector.project(grads, model_id)

        assert projected_grads.shape == (self.proj_max_batch_size, self.proj_dim)


if __name__ == "__main__":
    unittest.main()
