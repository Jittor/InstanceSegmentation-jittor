# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from detectron.modeling import registry
import  jittor as jt 
from jittor import nn,Module,init

@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        init.gauss_(self.cls_score.weight, mean=0, std=0.01)
        init.constant_(self.cls_score.bias, 0)

        init.gauss_(self.bbox_pred.weight, mean=0, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def execute(self, x):
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        init.gauss_(self.cls_score.weight, std=0.01)
        init.gauss_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            init.constant_(l.bias, 0)

    def execute(self, x):
        if x.ndim == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.reshape(x.shape[0], -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
