# rustputo

This Python library implements the Caputo fractional derivative as described in Zhang et al. (2021). The core functionality is provided via a Rust extension module for high performance, and can be used directly from Python.

## Features

- Caputo fractional derivative computation
- Integration with NumPy arrays
- Rust-powered backend for speed

## Installation

Install via pip (requires Rust toolchain):

```sh
pip install .
```

## Usage

```python
import rustputo.model as model

# Example usage
result = model.caputo_derivative(array, order)
```

## Reference

Zhang, et al. (2021). [Title of the paper]. *Journal Name*, Volume(Issue), pages. [DOI or link]

## License

MIT License.


## TODO:

Need to transfer to setuptools-rust.