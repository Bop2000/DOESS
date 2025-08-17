# DOESS
Learning to steer quantum many-body dynamics with artificial intelligence. Data-driven evOlutionary approach to Explore the Sequence Space (DOESS), pipeline showing below:
<p align="center">
  <img src="assets/doess.png" alt="DOESS" width="800">
</p>


## Pseudocode
<p align="center">
  <img src="assets/pseudo.png" alt="Pseudocode" width="700">
</p>


# System Requirements
## Hardware requirements
`DOESS` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *Linux* and *Windows*. The package has been tested on the following systems:
+ Linux: Ubuntu 18.04
+ Windows: Windows 10

### Python Dependencies
`DOESS` mainly depends on the Python scientific stack.

```
dependencies = [
    "numpy>=1.19.5",
    "pandas>=1.4.4",
    "matplotlib>=3.6.3",
    "seaborn>=0.12.2",
    "scikit-learn>=1.2.2",
    "scipy>=1.10.1",
    "tensorflow>=2.5.0",
    "pytest>=7.3.1",
    "tqdm>=4.65.0",
    "openpyxl>=3.1.2",
]
```

## Installation

`DOESS` requires `python>=3.8`. Installation of TensorFlow and Keras with CUDA support is strongly recommended. It typically takes a few minutes to finish the installation on a `normal` desktop computer.

To install DOESS, run:

```bash
pip install git+https://github.com/Bop2000/DOESS.git
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone https://github.com/Bop2000/DOESS
cd DOESS
pip install -e .
```

## Running Tests

To run tests for DOESS, execute the following command in the project root directory:

```bash
python -m pytest -m "not slow"
```


## DOESS optimization

The demo notebook with code execution history can be found in folder `notebook`.
Here's a detailed example of how to use DOESS:

```python
xxx
```

## License

The source code is released under the MIT license, as presented in [here](LICENSE).
