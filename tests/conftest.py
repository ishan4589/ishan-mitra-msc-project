# I used Claude Code to help implement and package the shared test fixtures.

"""Shared pytest fixtures."""

import pytest


@pytest.fixture(scope="session")
def configured():
    """Global numerical configuration, applied once per session.

    The project uses float64 to reduce round-off in verification and differentiated
    solver checks, so tests that depend on global numerical settings request this
    fixture.
    """
    from neuralphysics import configure

    return configure()
