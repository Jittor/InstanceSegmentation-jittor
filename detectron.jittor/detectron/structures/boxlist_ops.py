# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import jittor as jt 

from .bounding_box import BoxList

from detectron.layers import nms as _box_nms
from detectron.layers import ml_nms as _box_ml_nms

from jittor.utils.nvtx import nvtx_scope

# @nvtx_scope("boxlist_nms")
def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0 or len(boxlist)==0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)

def boxlist_ml_nms(boxlist, nms_thresh, max_proposals=-1,
                   score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    scores = boxlist.get_field(score_field)
    labels = boxlist.get_field(label_field)
    keep = _box_ml_nms(boxes, scores, labels.float(), nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)



def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = jt.maximum(box1[:, :2].unsqueeze(1), box2[:, :2])  # [N,M,2]
    rb = jt.minimum(box1[:, 2:].unsqueeze(1), box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min_v=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1.unsqueeze(1) + area2 - inter)
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return jt.contrib.concat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)
    if len(bboxes)==0:
        return BoxList(jt.empty((0,4)), (0,0), mode='xyxy')
    
    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)
    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)
    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)
    return cat_boxes


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_partly_overlap(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = jt.maximum(box1[:, :2].unsqueeze(1), box2[:, :2])  # [N,M,2]
    rb = jt.minimum(box1[:, 2:].unsqueeze(1), box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min_v=0,max_v=999999)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:].unsqueeze(1) + area2 - inter)
    overlap = iou > 0
    not_complete_overlap = (inter - area1[:].unsqueeze(1)) * (inter - area2[ :].unsqueeze(0)) != 0
    partly_overlap = overlap * not_complete_overlap

    return partly_overlap

def boxlist_overlap(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = jt.maximum(box1[:, :2].unsqueeze(1), box2[:, :2])  # [N,M,2]
    rb = jt.minimum(box1[:, 2:].unsqueeze(1), box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min_v=0,max_v=999999)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:].unsqueeze(1) + area2 - inter)
    overlap = iou > 0

    return overlap


def boxes_to_masks(boxes, h, w, padding = 0.0):
    n = boxes.shape[0]
    boxes = boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    b_w = x2 - x1
    b_h = y2 - y1
    x1 = jt.clamp(x1-1-b_w*padding, min_v=0)
    x2 = jt.clamp(x2+1+b_w*padding, max_v=w)
    y1 = jt.clamp(y1-1-b_h*padding, min_v=0)
    y2 = jt.clamp(y2+1+b_h*padding, max_v=h)

    rows = jt.arange(w,dtype=x1.dtype).view(1, 1, -1).expand((n, h, w))
    cols = jt.arange(h,dtype=x1.dtype).view(1, -1, 1).expand((n, h, w))

    masks_left = rows >= x1.view(-1, 1, 1)
    masks_right = rows < x2.view(-1, 1, 1)
    masks_up = cols >= y1.view(-1, 1, 1)
    masks_down = cols < y2.view(-1, 1, 1)

    masks = masks_left * masks_right * masks_up * masks_down

    return masks

def crop_by_box(masks, box, padding = 0.0):
    n, h, w = masks.size()

    b_w = box[:, 2] - box[:, 0]
    b_h = box[:, 3] - box[:, 1]
    x1 = jt.clamp(box[:, 0] - b_w * padding - 1, min_v=0)
    x2 = jt.clamp(box[:, 2] + b_w * padding + 1, max_v=w-1)
    y1 = jt.clamp(box[:, 1] - b_h * padding - 1, min_v=0)
    y2 = jt.clamp(box[:, 3] + b_h * padding + 1, max_v=h-1)

    rows = jt.arange(w, dtype=x1.dtype).view(1, 1, -1).expand((n, h, w))
    cols = jt.arange(h,  dtype=x1.dtype).view(1, -1, 1).expand((n, h, w))

    masks_left = rows >= x1.view(n, 1, 1)
    masks_right = rows < x2.view(n, 1, 1)
    masks_up = cols >= y1.view(n, 1, 1)
    masks_down = cols < y2.view(n, 1, 1)

    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.float(), crop_mask