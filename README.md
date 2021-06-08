# liegroups
Python implementation of SO2, SE2, SO3, and SE3 matrix Lie groups using numpy or PyTorch.

## Install from github
```bash
pip install git+https://github.com/neutrinoyu/liegroups.git
```

## Installation (local)
To install, `cd` into the repository directory (the one with `setup.py`) and run:
```bash
pip install .
```
or
```bash
pip install -e .
```
The `-e` flag tells pip to install the package in-place, which lets you make changes to the code without having to reinstall every time. *Do not do this on shared workstations!*

## Testing
Ensure you have `pytest` installed on your system, or install it using `conda install pytest` or `pip install pytest`. Then run `pytest` in the repository directory.

## Usage

Access the numpy implementation using something like
```python
from liegroups import SE3
```