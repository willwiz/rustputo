# rustputo

This Python library implements the Caputo fractional derivative as described in Zhang et al. (2021). The core functionality is provided via a Rust extension module for high performance, and can be used directly from Python. Some of the modern constitutive models for soft tissues are also implemented here.

## Features

- Caputo fractional derivative computation
- Integration with NumPy arrays
- Common constitutive models for soft tissues
- Rust-powered backend for speed
- Fully typed
- Some commonly used models
    - Neohookean
    - Holzapfel Ogden
    - Nordsletten
    - Isotropic exponential
    - Planer Linear Isotropic

## Installation

Install via pip (requires Rust toolchain):

```sh
pip install .
```

## Usage

See models.pyi

## Reference
To be added

Zhang, et al. (XXXX). [Title of the paper]. *Journal Name*, Volume(Issue), pages. [DOI or link]

## License

MIT License.
