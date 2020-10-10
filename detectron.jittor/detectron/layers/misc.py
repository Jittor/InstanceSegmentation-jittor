# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
helper class that supports empty tensors on some nn functions.

Ideally, add support directly in PyTorch to empty tensors in
those functions.

This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math
from jittor import nn,Module,init
from jittor.misc import  _ntuple
from jittor import Function
import jittor as jt



class _NewEmptyTensorOp(Function):
    def execute(self, x, new_shape):
        self.save_vars = x.shape
        return jt.empty(new_shape,str(x.dtype))

    def grad(self, grad):
        shape = self.save_vars
        return self.execute(grad, shape), None


class Conv2d(nn.Conv):
    def execute(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).execute(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp()(x, output_shape)


class ConvTranspose2d(nn.ConvTranspose):
    def execute(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).execute(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp()(x, output_shape)


class BatchNorm2d(nn.BatchNorm):
    def execute(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).execute(x)
        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp()(x, output_shape)


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    if input.numel() > 0:
        return nn.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp()(input, output_shape)

class InterpolateModule(nn.Module):
	"""
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

	def __init__(self, *args, **kwdargs):
		super().__init__()

		self.args = args
		self.kwdargs = kwdargs

	def execute(self, x):
		return nn.interpolate(x, *self.args, **self.kwdargs)