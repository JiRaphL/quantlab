import pytest

from quantlab.data.panel import make_synthetic_panel


@pytest.fixture
def synthetic_panel():
    return make_synthetic_panel(["AAA", "BBB", "CCC", "DDD"], periods=1500, seed=42)
