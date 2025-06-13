import pytest
import rustputo


def test_sum_as_string():
    assert rustputo.sum_as_string(1, 1) == "2"
