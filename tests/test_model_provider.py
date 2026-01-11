import pytest
from ocr_benchmark.models import get_model_provider


def test_ground_truth_returns_none():
    assert get_model_provider("ground-truth") is None


def test_unknown_provider_raises():
    with pytest.raises(NotImplementedError):
        get_model_provider("this-does-not-exist")
