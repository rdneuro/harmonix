import numpy as np
import pytest

from harmonix.criticality import (branching_parameter, distance_to_criticality,
                                  kuramoto_order_parameter, metastability,
                                  numerical_abscissa, relaxation_time,
                                  spectral_abscissa, spectral_gap,
                                  working_point_spectral)
from harmonix.models import HopfModal


def test_abscissa_tracks_bifurcation(demo_basis):
    prev = -np.inf
    for a in (-0.1, -0.02, -0.001):
        absc = spectral_abscissa(HopfModal(demo_basis, bifurcation=a, coupling=0.0))
        assert absc < 0
        assert absc > prev                                   # rises toward 0
        prev = absc


def test_relaxation_time_diverges_near_critical(demo_basis):
    slow = relaxation_time(HopfModal(demo_basis, bifurcation=-0.001, coupling=0.0))
    fast = relaxation_time(HopfModal(demo_basis, bifurcation=-0.1, coupling=0.0))
    assert slow > fast


def test_distance_to_criticality_positive_when_stable(demo_basis):
    assert distance_to_criticality(HopfModal(demo_basis, bifurcation=-0.05)) > 0


def test_spectral_gap_and_numerical_abscissa(demo_basis):
    h = HopfModal(demo_basis, bifurcation=-0.02, coupling=0.1)
    assert spectral_gap(h) >= 0
    assert np.isfinite(numerical_abscissa(h))


def test_kuramoto_bounded():
    phases = np.random.default_rng(0).uniform(-np.pi, np.pi, (30, 100))
    R = kuramoto_order_parameter(phases)
    assert np.all((R >= 0) & (R <= 1.0 + 1e-9))


def test_working_point_returns_stable(demo_basis):
    def make(G):
        return HopfModal(demo_basis, bifurcation=-0.05, coupling=G)
    bestG, absc = working_point_spectral(make, np.linspace(0, 0.2, 11))
    assert np.isfinite(bestG)
    assert absc.shape == (11,)
