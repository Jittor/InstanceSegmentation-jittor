#coding=utf-8
import requests
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import glob
import os
import math

import pickle

os.environ['use_mkl']='0'

def get_config_root_path():
    ''' Path to configs for unit tests '''
    # cur_file_dir is root/tests/env_tests
    cur_file_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    ret = os.path.dirname(cur_file_dir)
    ret = os.path.join(ret, "configs")
    return ret

def get_config_files():
    root = get_config_root_path()
    files = glob.glob(os.path.join(root,'*/*.yaml'))
    return files 


def remove_tmp():
    root = '/home/lxl/tmp/'
    files = glob.glob(root+'*.pkl')
    for f in files:
        os.remove(f)
    
def load(img_f=None):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    if img_f is None:
        img_f = 'test.jpg'
    pil_image = Image.open(img_f).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def check_array(a, b):
    rtol = 5e-2
    atol = 1e-3
    err = np.abs(a-b)
    tol = atol + rtol * np.abs(b)
    is_error = np.logical_or( err > tol, (a>=-1e-5)!=(b>=-1e-5))
    index = np.where(is_error)
    assert len(index)>0
    if len(index[0]) == 0:
        return 0

    err_rate = is_error.mean()
    return err_rate
        
def result_diff(prediction):
    root = '/home/lxl/tmp/'
    data = {'bbox':prediction.bbox}
    data.update(prediction.extra_fields)
    files = glob.glob(root+'*.pkl')
    print(len(files),len(data))
    #assert len(files)==0 or len(files)==len(data)
    for k,d in data.items():
        if hasattr(d,'cpu'):
            d = d.cpu()
        if hasattr(d,'numpy'):
            d = d.detach().numpy()
        filename = root+k+'.pkl'
        if os.path.exists(filename):
            dd = pickle.load(open(filename,'rb'))
            err_rate = check_array(dd,d)
            if err_rate>0.01 or dd.shape[0]==0:
                print(k,err_rate)
                print('torch',dd)
                print('jittor',d)
                #assert False,k
            #print(k,'shape is ',d.shape,'Good')
        else:
            pickle.dump(d,open(filename,'wb'))

def run_model(config_file,img_f=None):
    original_image = load(img_f)
    from detectron.config import cfg
    from detectron.modeling.detector import build_detection_model
    from detectron.utils.checkpoint import DetectronCheckpointer
    from detectron.structures.image_list import to_image_list
    from detectron.modeling.roi_heads.mask_head.inference import Masker

    from jittor import transform as T
    from jittor import nn
    import jittor as jt
    from jittor_utils import auto_diff
    
    jt.flags.use_cuda=1
    confidence_threshold = 0.0

    cfg.merge_from_file(config_file)
    model = build_detection_model(cfg)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir = cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    
    name = config_file.split('/')[-1].split('.')[0]
    # hook = auto_diff.Hook(name)
    # hook.hook_module(model)
    model.eval()

    class Resize(object):
        def __init__(self, min_size, max_size):
            self.min_size = min_size
            self.max_size = max_size

        # modified from torchvision to add support for max size
        def get_size(self, image_size):
            w, h = image_size
            size = self.min_size
            max_size = self.max_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def __call__(self, image):
            size = self.get_size(image.size)
            image = T.resize(image, size)
            return image

    def build_transform():
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.ImageNormalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform
    
    transforms = build_transform()
    image = transforms(original_image)
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
    predictions = model(image_list)

    predictions = predictions[0]
    if predictions.has_field("mask_scores"):
        scores = predictions.get_field("mask_scores")
    else:
        scores = predictions.get_field("scores")
    
    keep = jt.nonzero(scores>confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    idx,_ = jt.argsort(scores,0, descending=True)
    predictions =  predictions[idx]

    result_diff(predictions)
    

def run_torch_model(config_file,img_f=None):
    original_image = load(img_f)
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.modeling.detector import build_detection_model
    from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
    from maskrcnn_benchmark.structures.image_list import to_image_list
    from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker

    import torch
    from torchvision import transforms as T
    from torchvision.transforms import functional as F
    import jittor as jt
    from jittor_utils import auto_diff
    cfg.merge_from_file(config_file)
    model = build_detection_model(cfg)
    confidence_threshold = 0.0


    checkpointer = DetectronCheckpointer(cfg, model, save_dir = cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    
    name = config_file.split('/')[-1].split('.')[0]
    # hook = auto_diff.Hook(name)
    # hook.hook_module(model)
    model.eval()

    class Resize(object):
        def __init__(self, min_size, max_size):
            self.min_size = min_size
            self.max_size = max_size

        # modified from torchvision to add support for max size
        def get_size(self, image_size):
            w, h = image_size
            size = self.min_size
            max_size = self.max_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def __call__(self, image):
            size = self.get_size(image.size)
            image = F.resize(image, size)
            return image

    def build_transform():
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform
    
    transforms = build_transform()
    image = transforms(original_image)
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
    predictions = model(image_list)

    predictions = predictions[0]

    if predictions.has_field("mask_scores"):
        scores = predictions.get_field("mask_scores")
    else:
        scores = predictions.get_field("scores")
    
    keep = torch.nonzero(scores>confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _,idx = torch.sort(scores,0, descending=True)
    predictions =  predictions[idx]

    result_diff(predictions)


def run_fcos_model(config_file):
    original_image = load()
    from fcos_core.config import cfg
    from fcos_core.modeling.detector import build_detection_model
    from fcos_core.utils.checkpoint import DetectronCheckpointer
    from fcos_core.structures.image_list import to_image_list
    from fcos_core.modeling.roi_heads.mask_head.inference import Masker

    import torch
    from torchvision import transforms as T
    from torchvision.transforms import functional as F
    import jittor as jt
    from jittor_utils import auto_diff
    cfg.merge_from_file(config_file)
    model = build_detection_model(cfg)
    model = model.cuda()
    confidence_threshold = 0.0


    checkpointer = DetectronCheckpointer(cfg, model, save_dir = cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    
    name = config_file.split('/')[-1].split('.')[0]
    #hook = auto_diff.Hook(name)
    #hook.hook_module(model)
    model.eval()

    class Resize(object):
        def __init__(self, min_size, max_size):
            self.min_size = min_size
            self.max_size = max_size

        # modified from torchvision to add support for max size
        def get_size(self, image_size):
            w, h = image_size
            size = self.min_size
            max_size = self.max_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def __call__(self, image):
            size = self.get_size(image.size)
            image = F.resize(image, size)
            return image

    def build_transform():
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform
    
    transforms = build_transform()
    image = transforms(original_image)
    image = image.cuda()
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
    predictions = model(image_list)

    predictions = predictions[0]

    if predictions.has_field("mask_scores"):
        scores = predictions.get_field("mask_scores")
    else:
        scores = predictions.get_field("scores")
    
    keep = torch.nonzero(scores>confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _,idx = torch.sort(scores,0, descending=True)
    predictions =  predictions[idx]

    result_diff(predictions)


def run_inference(config_file):
    import jittor as jt

    from detectron.config import cfg
    from detectron.modeling.detector import build_detection_model
    from detectron.utils.checkpoint import DetectronCheckpointer
    from detectron.data import make_data_loader
    from detectron.engine.inference import inference
    from detectron.utils.logger import setup_logger

    jt.flags.use_cuda=1
    #jt.cudnn.set_algorithm_cache_size = 100000

    cfg.merge_from_file(config_file)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir)
    model = build_detection_model(cfg)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False)
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.EMBED_MASK_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            cfg = cfg
        )

def run_torch_inference(config_file):
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.modeling.detector import build_detection_model
    from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
    from maskrcnn_benchmark.data import make_data_loader
    from maskrcnn_benchmark.engine.inference import inference
    from maskrcnn_benchmark.utils.logger import setup_logger

    cfg.merge_from_file(config_file)
    cfg.freeze()

    save_dir = ""
    model = build_detection_model(cfg)
    model = model.cuda()

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False)
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            'coco_2014_minival',
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        )

def run_all_models():
    #config_files = sorted(get_config_files())
    start = 7
    start = 5


    config_files = [
        '/home/lxl/jittor/detectron.jittor/configs/maskrcnn_benchmark/e2e_faster_rcnn_R_50_C4_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/maskrcnn_benchmark/e2e_faster_rcnn_R_50_FPN_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/maskrcnn_benchmark/e2e_faster_rcnn_R_101_FPN_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/maskrcnn_benchmark/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/maskrcnn_benchmark/e2e_mask_rcnn_R_50_C4_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/maskrcnn_benchmark/e2e_mask_rcnn_R_50_FPN_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/maskrcnn_benchmark/e2e_mask_rcnn_R_101_FPN_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/maskrcnn_benchmark/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/maskscoring_rcnn/e2e_ms_rcnn_R_50_FPN_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/maskscoring_rcnn/e2e_ms_rcnn_R_101_FPN_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/fcos/fcos_bn_bs16_MNV2_FPN_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/fcos/fcos_R_50_FPN_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/fcos/fcos_imprv_R_101_FPN_2x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/fcos/fcos_imprv_X_101_64x4d_FPN_2x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/fcos/fcos_R_101_FPN_2x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/fcos/fcos_X_101_64x4d_FPN_2x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/embedmask/embed_mask_R50_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/embedmask/embed_mask_R101_1x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/embedmask/embed_mask_R101_ms_3x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/centermask/centermask_R_50_FPN_ms_3x.yaml',
        '/home/lxl/jittor/detectron.jittor/configs/centermask/centermask_R_101_FPN_ms_2x.yaml',
    ][start:start+1]
    for f in config_files:
        print(f)
        #run_torch_inference(f)
        run_inference(f)
        # img_f = '/home/lxl/dataset/coco/images/val2014/COCO_val2014_000000135411.jpg'
        # remove_tmp()
        # run_torch_model(f,img_f)
        # run_model(f,img_f)
        # for img_f in glob.glob('/home/lxl/dataset/coco/images/val2014/*'):
        #     print(img_f)
        #     remove_tmp()
        #     run_torch_model(f,img_f)
        #     run_model(f,img_f)


if __name__ == '__main__':
    run_all_models()

