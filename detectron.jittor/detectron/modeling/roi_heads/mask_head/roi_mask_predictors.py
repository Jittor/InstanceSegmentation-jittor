# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import jittor as jt 
from jittor import nn,Module,init

from detectron.layers import Conv2d
from detectron.layers import ConvTranspose2d
from detectron.modeling import registry


@registry.ROI_MASK_PREDICTOR.register("MaskRCNNC4Predictor")
class MaskRCNNC4Predictor(Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for param in self.parameters():
            name = param.name()
            if "bias" in name:
                init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def execute(self, x):
        #print('predictor 1',jt.mean(x),jt.sum(x),x[0])
        x = self.conv5_mask(x)
        #print('predictor 2',jt.mean(x),jt.sum(x),x[0])

        x = nn.relu(x)
        #print('predictor 3',jt.mean(x),jt.sum(x),x[0])

        x = self.mask_fcn_logits(x)
        #print('predictor 4',jt.mean(x),jt.sum(x),x[0])

        return x


@registry.ROI_MASK_PREDICTOR.register("MaskRCNNConv1x1Predictor")
class MaskRCNNConv1x1Predictor(Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNConv1x1Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_inputs = in_channels

        self.mask_fcn_logits = Conv2d(num_inputs, num_classes, 1, 1, 0)

        for param in self.parameters():

            name = param.name()
            if "bias" in name:
                init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def execute(self, x):
        return self.mask_fcn_logits(x)


def make_roi_mask_predictor(cfg, in_channels):
    func = registry.ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg, in_channels)
