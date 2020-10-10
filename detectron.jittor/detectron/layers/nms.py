# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C

# Only valid with fp32 inputs - give AMP the hint
from jittor import nms as _nms
import jittor as jt
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""

from jittor.utils.nvtx import nvtx_scope
# @nvtx_scope("nms")
def nms(dets,scores,threshold):
    dets = jt.contrib.concat([dets,scores.unsqueeze(-1)],dim=-1)
    return _nms(dets, threshold)


# @nvtx_scope("_ml_nms")
def _ml_nms(dets,thresh):
    '''
      dets jt.array [x1,y1,x2,y2,score]
      x(:,0)->x1,x(:,1)->y1,x(:,2)->x2,x(:,3)->y2,x(:,4)->score
    '''
    threshold = str(thresh)
    tmp = dets[:,4]
    order,_ = jt.argsort(tmp,descending=True)
    dets = dets[order]

    s_1 = '(@x(j,2)-@x(j,0)+1)*(@x(j,3)-@x(j,1)+1)'
    s_2 = '(@x(i,2)-@x(i,0)+1)*(@x(i,3)-@x(i,1)+1)'
    s_inter_w = 'max((Tx)0,min(@x(j,2),@x(i,2))-max(@x(j,0),@x(i,0))+1)'
    s_inter_h = 'max((Tx)0,min(@x(j,3),@x(i,3))-max(@x(j,1),@x(i,1))+1)'
    s_inter = s_inter_h+'*'+s_inter_w
    iou = s_inter + '/(' + s_1 +'+' + s_2 + '-' + s_inter + ')'
    fail_cond = '('+iou+'>'+threshold+')&&(@x(i,5)==@x(j,5))'
    selected = jt.candidate(dets, fail_cond)
    return order[selected]

def ml_nms(boxes,scores,labels,threshold):
    dets = jt.contrib.concat([boxes,scores.unsqueeze(1),labels.unsqueeze(1)],dim=1)
    return _ml_nms(dets,threshold)




