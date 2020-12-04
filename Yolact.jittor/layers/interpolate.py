from jittor import nn


def interpolate(X,size=None,scale_factor=None,mode='bilinear',align_corners=False):
    if scale_factor is not None:
        size = [X.shape[-2]*scale_factor,X.shape[-1]*scale_factor]
    if isinstance(size,int):
        size = (size,size)
    return nn.resize(X,size,mode,align_corners)

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
		return interpolate(x, *self.args, **self.kwdargs)
