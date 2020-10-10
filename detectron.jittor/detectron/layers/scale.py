import jittor as jt 
from jittor import nn


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = jt.array([init_value]).float()

    def execute(self, input):
        return input * self.scale
