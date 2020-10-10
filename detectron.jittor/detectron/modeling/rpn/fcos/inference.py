# Modified by Youngwan Lee (ETRI). All Rights Reserved.
import jittor as jt
from jittor import nn

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from detectron.modeling.box_coder import BoxCoder
from detectron.modeling.utils import cat
from detectron.structures.bounding_box import BoxList
from detectron.structures.boxlist_ops import cat_boxlist
# from detectron.structures.boxlist_ops import boxlist_nms
from detectron.structures.boxlist_ops import boxlist_ml_nms
from detectron.structures.boxlist_ops import remove_small_boxes


class FCOSPostProcessor(nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh,
                 fpn_post_nms_top_n, min_size, num_classes, dense_points,bbox_aug_enabled,is_sqrt):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.dense_points = dense_points
        self.bbox_aug_enabled = bbox_aug_enabled
        self.is_sqrt = is_sqrt

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, self.num_classes - 1).sigmoid()
        box_regression = box_regression.view(N, self.dense_points * 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, self.dense_points, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max_v=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :].unsqueeze(2)
        results = []
        #print('forward_for_single_feature_map start',N)
        for i in range(N):
            #print(i)
            per_box_cls = box_cls[i]

            per_candidate_inds = candidate_inds[i]
            #print(per_candidate_inds.shape,per_box_cls.shape)
            # if per_candidate_inds.sum().item()>0:
            #    per_box_cls = per_box_cls[per_candidate_inds]
            # else:
            #    per_box_cls = jt.zeros((0,),dtype=per_box_cls.dtype)

            #print(per_candidate_inds.shape,jt.sum(per_candidate_inds))
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] 
            # if per_candidate_nonzeros.numel()>0:
            #     per_class = per_candidate_nonzeros[:, 1] + 1
            per_class = per_candidate_nonzeros[:, 1] + 1
            #print(per_candidate_nonzeros.shape)

            
            per_box_regression = box_regression[i]
            #print('GG',per_box_loc.numel(),per_box_loc.shape)
            # if per_box_loc.numel()>0:
            #     per_box_regression = per_box_regression[per_box_loc]
            #     per_locations = locations[per_box_loc]
            # else:
            #     shape = list(per_box_regression.shape)
            #     shape[0]=0
            #     per_box_regression = jt.zeros(shape,dtype=per_box_regression.dtype)
            #     shape = list(locations.shape)
            #     shape[0]=0
            #     per_locations = jt.zeros(shape,dtype=locations.dtype)

            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            #print('??')
            #print('per_box_cls1',per_box_cls.mean())

            per_pre_nms_top_n = pre_nms_top_n[i]

            #print('per_locations',jt.mean(per_locations))
            #print('per_box_regressions',jt.mean(per_box_regression))
            #print(per_pre_nms_top_n.item(),per_candidate_inds.sum().item())
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n.item(), sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
            
            #print('per_box_cls',per_box_cls.mean())
            #print('emmm',jt.mean(per_locations))
            #print('hhh',jt.mean(per_box_regression))
            # if per_box_loc.numel()>0:
            #     detections = jt.stack([
            #     per_locations[:, 0] - per_box_regression[:, 0],
            #     per_locations[:, 1] - per_box_regression[:, 1],
            #     per_locations[:, 0] + per_box_regression[:, 2],
            #     per_locations[:, 1] + per_box_regression[:, 3],
            # ], dim=1)
            # else:
            #     detections = jt.zeros((0,4),dtype=per_locations.dtype)
            detections = jt.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)
            #print('detections',jt.mean(detections),detections.shape)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            if self.is_sqrt:
                boxlist.add_field("scores", per_box_cls.sqrt())
            else:
                boxlist.add_field("scores", per_box_cls)
            #print('??',boxlist.get_field('scores'))
            if boxlist.bbox.numel()>0:
                boxlist = boxlist.clip_to_image(remove_empty=False)
                boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
            #print('Good')

        return results

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """

        gt_boxes = [target.copy_with_fields(['labels']) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("scores", jt.ones((len(gt_box))))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def execute(self, locations, box_cls, box_regression, centerness, image_sizes, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )
        '''
        
        for bb in sampled_boxes:
            for b in bb:
                print("Fcos Post sampled_boxes",jt.mean(b.bbox))
        '''
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        
        # for b in boxlists:
        #     print("fcos Post boxlists",b.bbox.mean())
        #     print('fcos Post boxlists',b.get_field('scores').mean())
        
        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists)

        
        if self.is_training() and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms

            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)

            #print('ml_nms',jt.mean(result.bbox))
            #print('scores',jt.mean(result.get_field("scores")))

            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = jt.kthvalue(
                    cls_scores,
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                #print(number_of_detections - self.fpn_post_nms_top_n + 1,self.fpn_post_nms_top_n,image_thresh)
                keep = cls_scores >= image_thresh.item()
                keep = jt.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(config, is_train):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    dense_points = config.MODEL.FCOS.DENSE_POINTS
    fpn_post_nms_top_n = config.MODEL.FCOS.POST_NMS_TOP_N_TRAIN #500

    if not is_train:
        fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED
    is_sqrt = config.MODEL.FCOS.IS_SQRT


    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        dense_points=dense_points,
        bbox_aug_enabled = bbox_aug_enabled,
        is_sqrt=is_sqrt)

    return box_selector
