# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import jittor as jt 
from jittor import nn,Module,init

from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from detectron.modeling import registry
from detectron.modeling.poolers import Pooler
from detectron.modeling.make_layers import make_conv3x3
from detectron.modeling.make_layers import SpatialAttention



registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        self.maskiou = cfg.MODEL.MASKIOU_ON

        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler
        

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            setattr(self,layer_name,module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def execute(self, x, proposals):
        #print('feature_extrac 0',x[0])
        x = self.pooler(x, proposals)
        #print('feature_extrac 1',x[0])
        roi_feature = x 
        for layer_name in self.blocks:
            #print('feature_extrac',i,x[0])
            x = nn.relu(getattr(self,layer_name)(x))
        if self.maskiou:
            return x,roi_feature
        return x


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNSpatialAttentionFeatureExtractor")
class MaskRCNNFPNSpatialAttentionFeatureExtractor(Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNSpatialAttentionFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        lvl_map_func = cfg.MODEL.ROI_MASK_HEAD.LEVEL_MAP_FUNCTION
        self.maskiou = cfg.MODEL.MASKIOU_ON
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            lvl_map_func=lvl_map_func
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        #spatial attention module
        self.spatialAtt = SpatialAttention()
        self.num_pooler = len(scales)

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            setattr(self,layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def execute(self, x, proposals):
        #for i in range(len(x)):
        #    print(jt.mean(x[i]))
        #print('------')
        #for i in range(len(proposals)):
        #    print(jt.mean(proposals[i].bbox))
        #print('----')
        x = self.pooler(x, proposals)
        #print(jt.mean(x))
        if self.maskiou:
            roi_feature = x
        #print(jt.mean(x))
        for layer_name in self.blocks:
            x = nn.relu(getattr(self, layer_name)(x))
        #print(jt.mean(x))
        #spatial attention
        if self.spatialAtt is not None:
            x = self.spatialAtt(x)
        #print(jt.mean(x))

        if self.maskiou:
            return x, roi_feature
        else:
            return x

def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
