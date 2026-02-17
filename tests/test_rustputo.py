# /// script
# requires-python = ">=3.14"
# dependencies = [rustputo]
# ///

import numpy as np

from rustputo.rust import axpy, lgres_mat, mult, sum_as_string


def test_mult() -> None:
    x = np.linspace(0, 1, 10, dtype=np.float64)
    y = np.full_like(x, 1.0)
    b = axpy(12.0, x, y)
    assert np.allclose(b, 12.0 * x + 1.0)
    mult(2.0, b)
    assert np.allclose(b, 2.0 * (12.0 * x + 1.0))


def test_sum_as_string() -> None:
    s = sum_as_string(1, 123)
    assert s == "124"
    assert isinstance(s, str)
    assert s + "okay" == "124okay"


def test_lgres_mat() -> None:
    mata = np.arange(9, dtype=np.float64).reshape((3, 3))
    mata = (mata.T + mata) / 2 + np.eye(3)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    res = lgres_mat(mata, v)
    assert np.allclose(mata @ res, v)


def main() -> None:
    x = np.linspace(0, 1, 10, dtype=np.float64)
    y = np.full_like(x, 1.0)
    b = axpy(12.0, x, y)
    print(b)
    mult(2.0, b)
    print(x)
    s = sum_as_string(1, 123)
    print(f"{1} + {123} = {s} is {type(s)} and {s + 'okay'}")
    mata = np.arange(9, dtype=np.float64).reshape((3, 3))
    mata = (mata.T + mata) / 2 + np.eye(3)
    print(mata)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    w = v @ v
    res_1 = np.linalg.solve(mata, v)
    print(np.linalg.inv(mata.T @ mata) @ v)
    res = lgres_mat(mata, v)
    print(mata @ res)
    print(res_1)
    print(res)
    print(axpy(1.0 / w, res, v))


if __name__ == "__main__":
    main()
