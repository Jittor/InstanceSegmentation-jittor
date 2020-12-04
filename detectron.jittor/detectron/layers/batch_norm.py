# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from jittor import nn,Module
import jittor as jt

class FrozenBatchNorm2d(Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.weight =  jt.ones(n)
        self.bias = jt.zeros(n)
        self.running_mean = jt.zeros(n)
        self.running_var = jt.ones(n)

    def execute(self, x):
        # Cast all fixed parameters to half() if necessary
        if str(x.dtype) == 'float16':
            self.weight = self.weight.float16()
            self.bias = self.bias.float16()
            self.running_mean = self.running_mean.float16()
            self.running_var = self.running_var.float16()

        if not hasattr(self, "_scale"):
            scale = self.weight / self.running_var.sqrt()
            bias = self.bias - self.running_mean * scale
            self._scale = scale.reshape(1, -1, 1, 1)
            self._bias = bias.reshape(1, -1, 1, 1)
        return x * self._scale + self._bias
