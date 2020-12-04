# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import jittor as jt 
from jittor import nn,Module,init

from detectron.modeling.box_coder import BoxCoder
from detectron.structures.bounding_box import BoxList
from detectron.structures.boxlist_ops import cat_boxlist
from detectron.structures.boxlist_ops import boxlist_nms
from detectron.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
import numpy as np
from .utils import permute_and_flatten
# II = 0

class RPNPostProcessor(Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
        fpn_post_nms_per_batch=True,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", jt.ones(len(gt_box)))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        # global II
        # import pickle
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).reshape(N, -1)
        # print('objectness',objectness.mean())

        objectness = objectness.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        # print('regression',box_regression.mean())

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        # print(pre_nms_top_n)
        #print('objectness',objectness)
        # objectness = jt.array(pickle.load(open(f'/home/lxl/objectness_0_{II}.pkl','rb')))
        
        # print(objectness.shape)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        # print(II,'topk',topk_idx.sum(),topk_idx.shape)
        batch_idx = jt.arange(N).unsqueeze(1)

        # pickle.dump(topk_idx.numpy(),open(f'/home/lxl/topk_idx_{II}_jt.pkl','wb'))
        # topk_idx_tmp = topk_idx.numpy()
        # batch_idx = jt.array(pickle.load(open(f'/home/lxl/batch_idx_{II}.pkl','rb')))
        # topk_idx = jt.array(pickle.load(open(f'/home/lxl/topk_idx_{II}.pkl','rb')))
        
        # err = np.abs(topk_idx_tmp-topk_idx.numpy())
        # print('Error!!!!!!!!!!!!!!!!',err.sum())
        # print(err.nonzero())

        
        #print('box_regression0',box_regression)
        #batch_idx = jt.index(topk_idx.shape,dim=0)
        box_regression = box_regression[batch_idx,topk_idx]
        #print('box_regression1',box_regression)

        image_shapes = [box.size for box in anchors]
        concat_anchors = jt.contrib.concat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]
        
        
        

        # box_regression = jt.array(pickle.load(open(f'/home/lxl/box_regression_{II}.pkl','rb')))
        # concat_anchors = jt.array(pickle.load(open(f'/home/lxl/concat_anchors_{II}.pkl','rb')))

        proposals = self.box_coder.decode(
            box_regression.reshape(-1, 4), concat_anchors.reshape(-1, 4)
        )

        proposals = proposals.reshape(N, -1, 4)

        # proposals = jt.array(pickle.load(open(f'/home/lxl/proposal_{II}.pkl','rb')))
        # objectness = jt.array(pickle.load(open(f'/home/lxl/objectness_{II}.pkl','rb')))
        # II+=1

        result = []
        for i in range(len(image_shapes)):
            proposal, score, im_shape = proposals[i], objectness[i], image_shapes[i] 
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            boxlist.add_field("objectness", score)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist = boxlist_nms(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)
        return result

    def execute(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        # import pickle
        # for i in range(len(anchors)):

        #     a = anchors[i][0]

        #     pickle.dump(a.bbox.numpy(),open(f'/home/lxl/anchor_{i}_jt.pkl','wb'))
        #     pickle.dump(objectness[i].numpy(),open(f'/home/lxl/objectness_{i}_jt.pkl','wb'))
        #     pickle.dump(box_regression[i].numpy(),open(f'/home/lxl/box_regression_{i}_jt.pkl','wb'))
        
        # for i in range(len(anchors)):
        #     anchors[i] = list(anchors[i])
        #     anchors[i][0].bbox = jt.array(pickle.load(open(f'/home/lxl/anchor_{i}_torch.pkl','rb')))
        #     objectness[i] = jt.array(pickle.load(open(f'/home/lxl/objectness_{i}_torch.pkl','rb')))
        #     box_regression[i] = jt.array(pickle.load(open(f'/home/lxl/box_regression_{i}_torch.pkl','rb')))
            
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        #print('sampled_boxes',sampled_boxes[0][0].bbox)

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        # boxlists[0].bbox = jt.array(pickle.load(open('/home/lxl/box_torch.pkl','rb')))
        # print('boxlists',boxlists[0].bbox,boxlists[0].bbox.mean())

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        if self.is_training() and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # NOTE: it should be per image, and not per batch. However, to be consistent 
        # with Detectron, the default is per batch (see Issue #672)
        if self.is_training() and self.fpn_post_nms_per_batch:
            objectness = jt.contrib.concat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = jt.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = jt.zeros(objectness.shape).bool()
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                # import pickle
                # pickle.dump(objectness.numpy(),open('/home/lxl/tmp_jt.pkl','wb'))
                # objectness = pickle.load(open('/home/lxl/tmp_torch.pkl','rb'))
                # objectness = jt.array(objectness)
                # print(objectness,objectness.mean())
                post_nms_top_n = min(self.fpn_post_nms_top_n, objectness.shape[0])
                _, inds_sorted = jt.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                # pickle.dump(inds_sorted.numpy(),open('/home/lxl/tmp_ind_jt.pkl','wb'))
                # inds_sorted = pickle.load(open('/home/lxl/tmp_ind_torch.pkl','rb'))
                # inds_sorted = jt.array(inds_sorted)

                # pickle.dump(boxlists[i].bbox.numpy(),open('/home/lxl/tmp_box_jt.pkl','wb'))
                # boxlists[i].bbox = pickle.load(open('/home/lxl/tmp_box_torch.pkl','rb'))
                # boxlists[i].bbox = jt.array(boxlists[i].bbox)
                # print(inds_sorted,inds_sorted.mean())
                boxlists[i] = boxlists[i][inds_sorted]
                # print(boxlists[i].bbox)
                
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    fpn_post_nms_per_batch = config.MODEL.RPN.FPN_POST_NMS_PER_BATCH
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        fpn_post_nms_per_batch=fpn_post_nms_per_batch,
    )
    return box_selector
