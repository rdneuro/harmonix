"""Shared fixtures for the harmonix test suite."""
import numpy as np
import pytest

from harmonix.bases import geometric_modes, icosphere


@pytest.fixture(scope="session")
def demo_mesh():
    return icosphere(3)


@pytest.fixture(scope="session")
def demo_basis(demo_mesh):
    V, F = demo_mesh
    return geometric_modes(V, F, n_modes=60)
