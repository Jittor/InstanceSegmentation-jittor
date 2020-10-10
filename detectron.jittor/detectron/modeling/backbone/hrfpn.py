import jittor as jt
from jittor import nn,init,Module

class HRFPN(Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize=None,
                 pooling='AVG',
                 share_conv=False,
                 conv_stride=1,
                 num_level=5,
                 with_checkpoint=False):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.share_conv = share_conv
        self.num_level = num_level
        self.reduction_conv = nn.Sequential(
            nn.Conv(in_channels=sum(in_channels),
                      out_channels=out_channels,
                      kernel_size=1),
        )

        if self.share_conv:
            self.fpn_conv = nn.Conv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3, 
                stride=conv_stride,
                padding=1,
            )
        else:
            self.fpn_conv = nn.ModuleList()
            for i in range(self.num_level):
                self.fpn_conv.append(nn.Conv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=conv_stride,
                    padding=1
                ))
        if pooling == 'MAX':
            self.pooling = 'maximum'
        else:
            self.pooling = 'mean'
        self.with_checkpoint = with_checkpoint

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv):
                nn.init.kaiming_normal_(m.weight,  a=1)
                nn.init.constant_(m.bias, 0)

    def execute(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            outs.append(nn.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = jt.contrib.concat(outs, dim=1)
        '''
        if out.requires_grad and self.with_checkpoint:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)
        '''
        out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_level):
            outs.append(nn.pool(out, kernel_size=2**i, stride=2**i,op=self.pooling))
        outputs = []
        if self.share_conv:
            for i in range(self.num_level):
                outputs.append(self.fpn_conv(outs[i]))
        else:
            for i in range(self.num_level):
                if not outs[i].is_stop_grad() and self.with_checkpoint:
                    tmp_out = checkpoint(self.fpn_conv[i], outs[i])
                else:
                    tmp_out = self.fpn_conv[i](outs[i])
                outputs.append(tmp_out)
        return tuple(outputs)