# I used Claude Code to help implement and package tests for utils.py.

"""Tests for neuralphysics.utils"""

import numpy as np
import torch

from neuralphysics.utils import configure, to_numpy

# ---- to_numpy ---------------------------------------------------------------


def test_to_numpy_passes_arrays_through_unchanged():
    """A NumPy input is returned as the same object rather than copied."""
    a = np.arange(6.0).reshape(2, 3)
    assert to_numpy(a) is a


def test_to_numpy_detaches_tensor_requiring_grad():
    """The case the docstring exists for: without .detach() this raises."""
    t = torch.ones(3, dtype=torch.float64, requires_grad=True)
    out = to_numpy(t * 2.0)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, np.full(3, 2.0))


# ---- configure --------------------------------------------------------------


def test_configure_makes_two_successive_draws_identical():
    """Reseeding reproduces the stream, for both
    generators."""
    configure(seed=1234)
    a_torch, a_numpy = torch.randn(5), np.random.rand(5)
    configure(seed=1234)
    b_torch, b_numpy = torch.randn(5), np.random.rand(5)
    assert torch.equal(a_torch, b_torch)
    assert np.array_equal(a_numpy, b_numpy)


def test_configure_different_seeds_differ():
    """Changing the seed changes the subsequent PyTorch draw."""
    configure(seed=1)
    a = torch.randn(5)
    configure(seed=2)
    assert not torch.equal(a, torch.randn(5))


def test_configure_restores_float64_for_the_rest_of_the_session():
    """Guard that the preceding float32 configuration does not leak into later tests."""
    configure(dtype=torch.float32)
    assert torch.get_default_dtype() is torch.float32
    configure()
    assert torch.get_default_dtype() is torch.float64
