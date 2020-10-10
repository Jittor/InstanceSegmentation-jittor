# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import jittor as jt 
from jittor import nn,Module,init
import math

from detectron.layers import ROIAlign

from .utils import cat


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = round(k_min,4)
        self.k_max = round(k_max,4)
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = jt.sqrt(cat([boxlist.area() for boxlist in boxlists]))


        # Eqn.(1) in FPN paper
        target_lvls = jt.floor(self.lvl0 + jt.log2(s / self.s0 + self.eps))
        target_lvls = jt.clamp(target_lvls, min_v=self.k_min, max_v=self.k_max)
        return target_lvls.int32() - self.k_min


class LevelMapperwithArea(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the Equation (2) in the CenterMask paper.
    """

    def __init__(self, k_min, k_max, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            eps (float)
        """
        self.k_min = round(k_min,4)
        self.k_max = round(k_max,4)
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        bbox_area = cat([boxlist.area() for boxlist in boxlists])
        img_area = cat([boxlist.image_area() for boxlist in boxlists])

        target_lvls = jt.ceil(self.k_max - jt.log2(img_area / bbox_area + self.eps))
        target_lvls = jt.clamp(target_lvls, min_v=self.k_min, max_v=self.k_max)
        return target_lvls.int32() - self.k_min


class Pooler(Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio,lvl_map_func='MASKRCNNLevelMapFunc'):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(*poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -math.log(scales[0],2.0)
        lvl_max = -math.log(scales[-1],2.0)
        self.map_levels = LevelMapper(lvl_min, lvl_max) if lvl_map_func == 'MASKRCNNLevelMapFunc' \
        else LevelMapperwithArea(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        #print('concat_boxes',concat_boxes)
        dtype =str(concat_boxes.dtype)
        ids = cat(
            [
                jt.full((len(b), 1), i, dtype=dtype)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = jt.contrib.concat([ids, concat_boxes], dim=1)
        return rois

    def execute(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        #print('boxes',boxes[0].bbox)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = rois.shape[0]
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype = str(x[0].dtype)
        result = jt.zeros(
            (num_rois, num_channels, output_size, output_size),
            dtype=dtype,
        )
        #print('rois',rois)
        i=0
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers.layers.values())):
            idx_in_level = jt.nonzero(levels == level).squeeze(1)
           
            rois_per_level = rois[idx_in_level]
            #print('idx_in_level',idx_in_level)
            #print('rois_per_level',rois_per_level)
            
            result[idx_in_level] = pooler(per_level_feature, rois_per_level).cast(dtype)
            #print(i,'---',pooler(per_level_feature, rois_per_level))

            i+=1
        return result


def make_pooler(cfg, head_name):
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO
    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
    )
    return pooler
