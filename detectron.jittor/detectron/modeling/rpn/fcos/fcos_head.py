# Modified by Youngwan Lee (ETRI). All Rights Reserved.
import math
import jittor as jt
from jittor import nn

from detectron.layers import Scale


class FCOSHead(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        setattr(self,'cls_tower', nn.Sequential(*cls_tower))
        setattr(self,'bbox_tower', nn.Sequential(*bbox_tower))
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS
        self.cls_logits = nn.Conv(
            in_channels, num_classes * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv(
            in_channels, 4 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv(
            in_channels, 1 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv):
                    nn.init.gauss_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
        self.cfg = cfg
        self.scales = nn.ModuleList(*[Scale(init_value=1.0) for _ in range(5)])

    def execute(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            

            if self.cfg.MODEL.RPN.FCOS_ONLY:
                centerness.append(self.centerness(cls_tower))
                bbox_reg.append(jt.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
                )))
                continue

            box_tower = self.bbox_tower(feature)
            '''
            centerness.append(self.centerness(box_tower))
            bbox_reg.append(jt.exp(self.scales[l](
                self.bbox_pred(box_tower)
            )))
            '''

            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = nn.relu(bbox_pred)
                if self.is_training():
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(jt.exp(bbox_pred))

        return logits, bbox_reg, centerness

class FCOSSharedHead(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSSharedHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.identity = cfg.MODEL.FCOS.RESIDUAL_CONNECTION
        shared_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            shared_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            shared_tower.append(nn.GroupNorm(32, in_channels))
            shared_tower.append(nn.ReLU())


        setattr(self,'shared_tower', nn.Sequential(*shared_tower))
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS
        self.cls_logits = nn.Conv(
            in_channels, num_classes * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv(
            in_channels, 4 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv(
            in_channels, 1 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.shared_tower, self.cls_logits, 
                          self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv):
                    nn.init.gauss_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList(*[Scale(init_value=1.0) for _ in range(5)])

    def execute(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            if self.identity:
                shared_tower = self.shared_tower(feature) + feature
            else:
                shared_tower = self.shared_tower(feature)
            logits.append(self.cls_logits(shared_tower))
            centerness.append(self.centerness(shared_tower))
            bbox_reg.append(jt.exp(self.scales[l](
                self.bbox_pred(shared_tower)
            )))

        return logits, bbox_reg, centerness