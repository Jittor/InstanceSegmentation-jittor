import jittor as jt 
from jittor import nn
from ..box_utils import decode, jaccard, index2d
from utils import timer

from data import cfg, mask_type

import numpy as np


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        
        self.use_cross_class_nms = False
        self.use_fast_nms = False

    def __call__(self, predictions, net):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        loc_data   = predictions['loc']
        conf_data  = predictions['conf']
        mask_data  = predictions['mask']
        prior_data = predictions['priors']

        proto_data = predictions['proto'] if 'proto' in predictions else None
        inst_data  = predictions['inst']  if 'inst'  in predictions else None

        out = []

        with timer.env('Detect'):
            batch_size = loc_data.shape[0]
            num_priors = prior_data.shape[0]

            conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(0,2, 1)
            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)
                with timer.env('self.detect'):
                    result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data)

                if result is not None and proto_data is not None:
                    result['proto'] = proto_data[batch_idx]

                out.append({'detection': result, 'net': net})
        
        return out


    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        with timer.env('Slices'):
            cur_scores = conf_preds[batch_idx, 1:]
            conf_scores = jt.max(cur_scores, dim=0)
        
            keep = (conf_scores > self.conf_thresh)
            keep = jt.where(keep)[0]

            scores = cur_scores[:, keep]
            boxes = decoded_boxes[keep]
            masks = mask_data[batch_idx, keep]

            if inst_data is not None:
                inst = inst_data[batch_idx, keep]
    
    
        if scores.shape[1] == 0:
            return None
        if self.use_fast_nms:
            if self.use_cross_class_nms:
                boxes, masks, classes, scores = self.cc_fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
            else:
                boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
        else:
            boxes, masks, classes, scores = self.traditional_nms(boxes, masks, scores, self.nms_thresh, self.conf_thresh)

            if self.use_cross_class_nms:
                print('Warning: Cross Class Traditional NMS is not implemented.')
        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}


    def cc_fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200):
        # Collapse all the classes into 1 
        classes,scores = scores.argmax(dim=0)

        idx,_ = scores.argsort(0, descending=True)
        idx = idx[:top_k]

        boxes_idx = boxes[idx]

        # Compute the pairwise IoU between the boxes
        iou = jaccard(boxes_idx, boxes_idx)
        
        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        iou = iou.triu_(diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the IoU matrix along the columns, each column will represent the maximum IoU
        # between this element and every element with a higher score than this element.
        iou_max = jt.max(iou, dim=0)

        # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
        # don't have a higher scoring box that would supress it in normal NMS.
        idx_out = idx[iou_max <= iou_threshold]
        
        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]

    def fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        idx,scores = scores.argsort(1, descending=True)

        idx = idx[:, :top_k]
        scores = scores[:, :top_k]
    
        num_classes, num_dets = idx.shape

        boxes = boxes[idx.view(-1)].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1)].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes)
        iou = iou.triu_(diagonal=1)
        iou_max= iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= iou_threshold)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        if second_threshold:
            keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = jt.arange(num_classes).unsqueeze(1).expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        
        #print('keep finish')
        # Only keep the top cfg.max_num_detections highest scores across all classes
        idx,scores  = jt.argsort(scores,dim=0, descending=True)
        #print('argsort finish')
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
        '''
        scores = scores[:cfg.max_num_detections]
        classes = classes[:cfg.max_num_detections]
        boxes = boxes[:cfg.max_num_detections]
        masks = masks[:cfg.max_num_detections]
        '''

        return boxes, masks, classes, scores

    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        import pyximport
        pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

        from utils.cython_nms import nms as cnms

        num_classes = scores.shape[0]

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size

        for _cls in range(num_classes):
            cls_scores = scores[_cls]
            conf_mask = cls_scores > conf_thresh
            idx = jt.arange(cls_scores.shape[0], device=boxes.device)

            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]

            if cls_scores.shape[0] == 0:
                continue
            
            preds = jt.contrib.concat([boxes[conf_mask], cls_scores[:, None]], dim=1).numpy()
            keep = cnms(preds, iou_threshold)
            keep = jt.array(keep).int64()

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])
        
        idx     = jt.contrib.concat(idx_lst, dim=0)
        classes = jt.contrib.concat(cls_lst, dim=0)
        scores  = jt.contrib.concat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores
