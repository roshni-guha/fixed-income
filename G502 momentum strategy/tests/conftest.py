"""
Pytest configuration and shared fixtures for G502 momentum strategy tests.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_dates():
    """Generate sample business day dates."""
    return pd.date_range(start="2020-01-01", periods=500, freq="B")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42
