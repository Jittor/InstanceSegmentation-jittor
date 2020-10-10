# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import jittor as jt

# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    diff = jt.abs(input - target)
    less_than_one = (diff<1.0).float32()
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    if size_average:
        return loss.mean()
    return loss.sum()
