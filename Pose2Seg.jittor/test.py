import argparse
import numpy as np
from tqdm import tqdm
import jittor as jt
from jittor import nn
import os.path as osp
from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask,COCOTEST
from pycocotools import mask as maskUtils
import time
import lib.transforms as translib
from lib.timer import Timers

from modeling.resnet import resnet50FPN
from modeling.affine_align import affine_align_gpu
from modeling.seg_module import resnet10units
from modeling.core import PoseAlign
from modeling.skeleton_feat import genSkeletons
import cv2

def collate_batch(batch):
    return batch

def test(model, dataset='cocoVal', logger=print,benchmark=False):    
    if dataset == 'OCHumanVal':
        ImageRoot = './data/OCHuman/images'
        AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json'
    elif dataset == 'OCHumanTest':
        ImageRoot = './data/OCHuman/images'
        AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json'
    elif dataset == 'cocoVal':
        ImageRoot = './data/coco2017/val2017'
        AnnoFile = './data/coco2017/annotations/person_keypoints_val2017_pose2seg.json'
    datainfos = COCOTEST(ImageRoot, AnnoFile, onlyperson=True, loadimg=True,is_test=True)
    datainfos.batch_size=1
    datainfos.num_workers = 1
    datainfos.collate_batch = collate_batch
    data_len = len(datainfos)
    #data_len = 1
    
    model.eval()
    
    results_segm = []
    imgIds = []
    start_time = time.time()
    outputs = []

    # jt.profiler.start(0, 0)

    # for i in tqdm(range(data_len)):
    for i,batch in tqdm(enumerate(datainfos)):
        #datainfos.display_worker_status()
        #if i>100:break
        # rawdata = datainfos[i]
        rawdata = batch[0]
        img = rawdata['data']
        image_id = rawdata['id']
        
        # height, width = img.shape[0:2]
        # gt_kpts = np.float32(rawdata['gt_keypoints']).transpose(0, 2, 1) # (N, 17, 3)
        # gt_segms = rawdata['segms']
        # gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])
        gt_kpts = rawdata['gt_kpts']
        gt_masks = rawdata['gt_masks']
        with jt.no_grad():
            output = model([img], [gt_kpts], [gt_masks],rawdata['test_input'])
        imgIds.append(image_id)
        #jt.display_memory_info()


        if benchmark:continue
        #outputs.append(output)
        for mask in output[0]:
            #print(np.sum(mask))
            maskencode = maskUtils.encode(np.asfortranarray(mask))
            maskencode['counts'] = maskencode['counts'].decode('ascii')
            results_segm.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "score": 1.0,
                    "segmentation": maskencode
                })
    jt.sync_all(True)

    # jt.profiler.stop()
    # jt.profiler.report()
    '''
    for output,image_id in zip(outputs,imgIds):
        for mask in output[0]:
            #print(np.sum(mask))
            maskencode = maskUtils.encode(np.asfortranarray(mask))
            maskencode['counts'] = maskencode['counts'].decode('ascii')
            results_segm.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "score": 1.0,
                    "segmentation": maskencode
                })
    '''
    # print(len(results_segm))
    end_time = time.time()
    print('fps',data_len/(end_time-start_time))

    
    
    if benchmark:return

    def do_eval_coco(image_ids, coco, results, flag):
        from pycocotools.cocoeval import COCOeval
        assert flag in ['bbox', 'segm', 'keypoints']
        # Evaluate
        coco_results = coco.loadRes(results)
        cocoEval = COCOeval(coco, coco_results, flag)
        cocoEval.params.imgIds = image_ids
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize() 
        return cocoEval
    
    cocoEval = do_eval_coco(imgIds, datainfos.COCO, results_segm, 'segm')
    logger('[POSE2SEG]          AP|.5|.75| S| M| L|    AR|.5|.75| S| M| L|')
    _str = '[segm_score] %s '%dataset
    for value in cocoEval.stats.tolist():
        _str += '%.3f '%value
    logger(_str)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pose2Seg Testing")
    parser.add_argument(
        "--weights",
        help="path to .pkl model weight",
        default='./weights/pose2seg_release.pkl',
        type=str,
    )
    parser.add_argument(
        "--coco",
        help="Do test on COCOPersons val set",
        action="store_true",
    )
    parser.add_argument(
        "--benchmark",
        help="Do fps test without warpaffine",
        action="store_true",
    )
    parser.add_argument(
        "--OCHuman",
        help="Do test on OCHuman val&test set",
        action="store_true",
    )
    
    args = parser.parse_args()
    jt.flags.use_cuda=1
    
    print('===========> loading model <===========')
    model = Pose2Seg()
    model.init(args.weights)
    model.benchmark = args.benchmark

    print('===========>   testing    <===========')
    if args.coco:
        test(model, dataset='cocoVal',benchmark = args.benchmark) 
    if args.OCHuman:
        test(model, dataset='OCHumanVal',benchmark =args.benchmark)
        test(model, dataset='OCHumanTest',benchmark =args.benchmark) 
