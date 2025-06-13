# ruff: noqa: S101
import unittest

import numpy as np
from rustputo.model import axpy, mult, sum_as_string


class TestRustPuto(unittest.TestCase):
    def test_axpy(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        result = axpy(2.0, x, y)
        expected = [6.0, 9.0, 12.0]
        assert (result == expected).all()

    def test_mult(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        mult(2.0, x)
        expected = [2.0, 4.0, 6.0]
        assert (x == expected).all()

    def test_sum_as_string(self) -> None:
        result = sum_as_string(1, 123)
        assert result == "124"
