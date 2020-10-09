from jittor import nn,Module
import jittor as jt
__all__ = ['resnet10units']

class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        #print("block1",x.mean(),x.sum())

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print("block2",out.mean(),out.sum())

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print("block3",out.mean(),out.sum())

        out = self.conv3(out)
        out = self.bn3(out)
        #print("block4",out.mean(),out.sum())

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        #print("block5",out.mean(),out.sum())

        return out


def interpolate(X,output_size=None,scale_factor=None,mode='nearest',align_corners=False):
    if scale_factor is not None:
        output_size = [X.shape[-2]*scale_factor,X.shape[-1]*scale_factor]
    if isinstance(output_size,int):
        output_size = (output_size,output_size)
    return nn.upsample(X,output_size,mode,align_corners)
    

class UpsamplingBilinear2d(Module):
    def __init__(self,scale_factor):
        self.scale_factor= scale_factor

    def execute(self,x):
        #print('--------------------',x.shape)
        return interpolate(x, scale_factor = self.scale_factor, mode='bilinear',align_corners=True)

    
class DeconvModule(Module):
    def __init__(self, in_channels, block, layers):
        super(DeconvModule, self).__init__()
        self.inplanes = 256
        self.conv1 = nn.Conv(in_channels, 256, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm(256)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 256, layers[0])
        
        self.upsample = UpsamplingBilinear2d(scale_factor=2)
        self.decoder1 = self._make_layer(block, 128, 1, stride = 1)
        self.pred_small = nn.Conv(128*block.expansion, 2, kernel_size=1, stride=1, bias=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def execute(self, x):
        #print(x.mean(),x.sum())
        x = self.conv1(x)
        #print(x.mean(),x.sum())
        x = self.bn1(x)
        #print(x.mean(),x.sum())
        x = self.relu(x)
        #print(x.mean(),x.sum())

        x = self.layer1(x)
        #print(x.mean(),x.sum())
        x = self.upsample(x)
        #print(x.mean(),x.sum())
        x = self.decoder1(x)
        #print(x.mean(),x.sum())
        
        x = self.pred_small(x)
        #print(x.mean(),x.sum())
        return x
    
def resnet10units(in_channel):
    return DeconvModule(in_channel, Bottleneck, [10])