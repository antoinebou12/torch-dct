# DCT (Discrete Cosine Transform) for **PyTorch**

[![Build Status](https://travis-ci.com/zh217/torch-dct.svg?branch=master)](https://travis-ci.com/zh217/torch-dct)
[![codecov](https://codecov.io/gh/zh217/torch-dct/branch/master/graph/badge.svg)](https://codecov.io/gh/zh217/torch-dct)
[![PyPI version](https://img.shields.io/pypi/v/torch-dct.svg)](https://pypi.python.org/pypi/torch-dct/)
[![PyPI versions](https://img.shields.io/pypi/pyversions/torch-dct.svg)](https://pypi.python.org/pypi/torch-dct/)
[![PyPI status](https://img.shields.io/pypi/status/torch-dct.svg)](https://pypi.python.org/pypi/torch-dct/)
[![License](https://img.shields.io/github/license/zh217/torch-dct.svg)](https://github.com/zh217/torch-dct/blob/master/LICENSE)

This library implements the Discrete Cosine Transform (DCT) using PyTorch’s built-in FFT operations, so backpropagation works seamlessly on both CPU and GPU.  
For background on the DCT and the algorithms used here, see the relevant article on [Wikipedia](https://en.wikipedia.org/wiki/Discrete_cosine_transform) and the classic paper by [Makhoul](https://ieeexplore.ieee.org/document/1163351). This [Stack Exchange thread](https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft) is also helpful.

### Implemented transforms

* **1-D** DCT-I and its inverse (a scaled DCT-I)  
* **1-D** DCT-II and its inverse (a scaled DCT-III)  
* **2-D** DCT-II and its inverse (a scaled DCT-III)  
* **3-D** DCT-II and its inverse (a scaled DCT-III)

---

## Installation

```bash
pip install torch-dct
````

Requires **PyTorch ≥ 0.4.1** (more recent versions are fully supported; earlier versions may work but are untested).

To run the test suite you’ll need `pytest` and `scipy`:

```bash
git clone https://github.com/zh217/torch-dct.git
cd torch-dct
pytest
```

---

## Quick start

```python
import torch
import torch_dct as dct

# 1-D example
x = torch.randn(200)
X = dct.dct(x)      # DCT-II along the last dimension
y = dct.idct(X)     # (scaled) inverse DCT-III

# Verify perfect reconstruction within numerical tolerance
assert torch.allclose(x, y, atol=1e-10)

# The DCT-I API mirrors this:
# X1 = dct.dct1(x); y1 = dct.idct1(X1)

# 2-D and 3-D versions:
# X2 = dct.dct_2d(img);  y2 = dct.idct_2d(X2)
# X3 = dct.dct_3d(vol);  y3 = dct.idct_3d(X3)
```
