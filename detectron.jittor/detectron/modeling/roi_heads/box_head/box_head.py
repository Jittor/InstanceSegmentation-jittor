# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from jittor import nn,Module
import jittor as jt
import time 

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class ROIBoxHead(Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def execute(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        #jt.sync_all()
        #print(5.1,1,time.asctime())
        #print('box_head start')
        if self.is_training():
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with jt.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
        #jt.sync_all()
        #print(5.1,2,time.asctime())
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads

        #print('box_head feature_extractor start')

        x = self.feature_extractor(features, proposals)
        # print(x)

        #print('box_head feature_extractor end')
        #jt.sync_all()
        #print(5.1,3,time.asctime())

        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        #print('box_head predictor end')

        # print(class_logits,box_regression)

        #jt.sync_all()
        #print(5.1,4,time.asctime())

        if not self.is_training():
            result = self.post_processor((class_logits, box_regression), proposals)
            #jt.sync_all()
            #print(5.1,5,time.asctime())
            return x, result, {}
        

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        #print('box_head end')

        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
