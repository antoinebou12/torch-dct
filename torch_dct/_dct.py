"""Discrete Cosine Transform utilities built on top of :mod:`torch.fft`."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch import nn


def dct1_rfft_impl(x: Tensor) -> Tensor:
    """Helper returning the real part of ``rfft`` for DCT-I."""
    return torch.fft.rfft(x, dim=1).real


def dct_fft_impl(v: Tensor) -> Tensor:
    """Helper returning real and imaginary parts of ``fft`` result."""
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def idct_irfft_impl(V: Tensor) -> Tensor:
    """Helper performing the inverse ``irfft`` for DCT calculations."""
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)



def dct1(x: Tensor, dim: int = -1) -> Tensor:
    """Compute the Type-I DCT along ``dim``."""
    x = torch.movedim(x, dim, -1)
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    x = torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)

    out = dct1_rfft_impl(x)
    out = out[..., : orig_shape[-1]]
    out = out.view(*orig_shape)
    return torch.movedim(out, -1, dim)


def idct1(X: Tensor, dim: int = -1) -> Tensor:
    """Inverse of :func:`dct1` along ``dim``."""
    n = X.shape[dim]
    return dct1(X, dim=dim) / (2 * (n - 1))


def dct(x: Tensor, norm: str | None = None, dim: int = -1) -> Tensor:
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x = torch.movedim(x, dim, -1)
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    V = torch.movedim(V, -1, dim)
    return V


def idct(X: Tensor, norm: str | None = None, dim: int = -1) -> Tensor:
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    X = torch.movedim(X, dim, -1)
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    x = x.view(*x_shape)
    return torch.movedim(x, -1, dim)


def dct_2d(x: Tensor, norm: str | None = None) -> Tensor:
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm, dim=-1)
    X2 = dct(X1, norm=norm, dim=-2)
    return X2


def idct_2d(X: Tensor, norm: str | None = None) -> Tensor:
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm, dim=-1)
    x2 = idct(x1, norm=norm, dim=-2)
    return x2


def dct_3d(x: Tensor, norm: str | None = None) -> Tensor:
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm, dim=-1)
    X2 = dct(X1, norm=norm, dim=-2)
    X3 = dct(X2, norm=norm, dim=-3)
    return X3


def idct_3d(X: Tensor, norm: str | None = None) -> Tensor:
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm, dim=-1)
    x2 = idct(x1, norm=norm, dim=-2)
    x3 = idct(x2, norm=norm, dim=-3)
    return x3


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)

if __name__ == '__main__':
    x = torch.Tensor(1000,4096)
    x.normal_(0,1)
    linear_dct = LinearDCT(4096, 'dct')
    error = torch.abs(dct(x) - linear_dct(x))
    assert error.max() < 1e-3, (error, error.max())
    linear_idct = LinearDCT(4096, 'idct')
    error = torch.abs(idct(x) - linear_idct(x))
    assert error.max() < 1e-3, (error, error.max())

