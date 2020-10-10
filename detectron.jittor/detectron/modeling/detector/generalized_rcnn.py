# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from jittor import nn
import jittor as jt

from detectron.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import time


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def eval(self):
        super(GeneralizedRCNN, self).eval()
        for v in self.__dict__.values():
            if isinstance(v, nn.Module):
                v.eval()
        
    def execute(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.is_training() and targets is None:
            raise ValueError("In training mode, targets should be passed")
        #print(3,time.asctime())
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        # print('backbone',jt.mean(features[0]))
        #jt.sync_all()
        #print(4,time.asctime())
        # print('Backbone',features[0].mean())

        proposals, proposal_losses = self.rpn(images, features, targets)
        # print('RPN',proposals[0].bbox,proposals[0].bbox.shape)


        #jt.sync_all()
        #print(5,time.asctime())
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
            #print('x',x)
            #print('result',result[0].bbox)
            #print('detector_losses',detector_losses)
        
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        #jt.sync_all()
        #print(6,time.asctime())

        if self.is_training():
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
