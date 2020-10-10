# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import jittor as jt 
from jittor import nn,Module
import time

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .maskiou_head.maskiou_head import build_roi_maskiou_head

class CombinedROIHeads(Module):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__()
        for k,m in heads:
            setattr(self,k,m)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            getattr(self,'mask').feature_extractor = getattr(self,'box').feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            getattr(self,'keypoint').feature_extractor = getattr(self,'box').feature_extractor

    def execute(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        #jt.sync_all()
        #print(5.1,time.asctime())
        #print('box start')
        x, detections, loss_box = getattr(self,'box')(features, proposals, targets)
        #print('box end')
        
        #jt.sync_all()
        #print(5.2,time.asctime())
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.is_training()
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            if not self.cfg.MODEL.MASKIOU_ON:
                x, detections, loss_mask = getattr(self,'mask')(mask_features, detections, targets)
                losses.update(loss_mask)
            else:
                x, detections, loss_mask, roi_feature, selected_mask, labels, maskiou_targets =getattr(self,'mask')(mask_features, detections, targets)
                losses.update(loss_mask)
                
                loss_maskiou, detections = getattr(self,'maskiou')(roi_feature, detections, selected_mask, labels, maskiou_targets)
                losses.update(loss_maskiou)

        #jt.sync_all()
        #print(5.3,time.asctime())
        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.is_training()
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = getattr(self,'keypoint')(keypoint_features, detections, targets)            
            losses.update(loss_keypoint)
        #jt.sync_all()
        #print(5.4,time.asctime())
        return x, detections, losses


class SingleTaskROIHeads(nn.Module):
    """
    Single task head (for masks).
    """
    def __init__(self, cfg, heads):
        super(SingleTaskROIHeads, self).__init__()
        for k,m in heads:
            setattr(self,k,m)
        self.cfg = cfg.clone()

    def execute(self, features, proposals, targets=None):
        losses = {}
        if self.cfg.MODEL.MASK_ON:
            mask_features = features

            if not self.cfg.MODEL.MASKIOU_ON:
                x, detections, loss_mask = getattr(self,'mask')(mask_features, proposals, targets)
                losses.update(loss_mask)
            else:
                x, detections, loss_mask, roi_feature, selected_mask, labels, maskiou_targets = getattr(self,'mask')(mask_features, proposals, targets)
                losses.update(loss_mask)
                
                loss_maskiou, detections =getattr(self,'maskiou')(roi_feature, detections, selected_mask, labels, maskiou_targets)
                losses.update(loss_maskiou)

        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []

    if cfg.MODEL.FCOS_MASK:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
        if cfg.MODEL.MASKIOU_ON:
            roi_heads.append(("maskiou", build_roi_maskiou_head(cfg, in_channels)))
        roi_heads = SingleTaskROIHeads(cfg, roi_heads)
        return roi_heads

    if cfg.MODEL.RETINANET_ON or cfg.MODEL.EMBED_MASK_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
        if cfg.MODEL.MASKIOU_ON:
            roi_heads.append(("maskiou", build_roi_maskiou_head(cfg,in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
