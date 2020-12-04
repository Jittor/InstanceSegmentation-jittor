# Instance Segmentation Model Zoo

实例分割作为计算机视觉的一项基本任务，在方方面面发挥着巨大的作用，本教程使用$Jittor$框架实现了实例分割模型库

实例分割模型库代码：https://github.com/Jittor/InstanceSegmentation-jittor

### 1 数据集

#### 1.1 COCO数据集下载

```bash
sudo wget -c http://images.cocodataset.org/zips/train2017.zip
sudo wget -c http://images.cocodataset.org/zips/val2017.zip
sudo wget -c http://images.cocodataset.org/zips/test2017.zip
sudo wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
sudo wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
```

将压缩包进行解压

```bash
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip
```

数据文件夹如下：

```
data  
├── coco2017
│   ├── annotations  
│   │   ├── instances_train2017.json 
│   │   ├── instances_val2017.json 
│   ├── images
|   │   ├── train2017  
│   │   │    ├── ####.jpg  
│   │   ├── val2017  
│   │   │    ├── ####.jpg  

```

#### 1.2 数据集路径配置

**Pose2Seg**:

下载转换好的pose2seg.json:

- [COCOPersons Train Annotation (person_keypoints_train2017_pose2seg.json) [166MB]](https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_train2017_pose2seg.json)
- [COCOPersons Val Annotation (person_keypoints_val2017_pose2seg.json) [7MB]](https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_val2017_pose2seg.json)

下载OCHuman:

- [images [667MB] & annotations](https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman)

数据格式：

```
data  
├── coco2017
│   ├── annotations  
│   │   ├── person_keypoints_train2017_pose2seg.json 
│   │   ├── person_keypoints_val2017_pose2seg.json 
│   ├── train2017  
│   │   ├── ####.jpg  
│   ├── val2017  
│   │   ├── ####.jpg  
├── OCHuman 
│   ├── annotations  
│   │   ├── ochuman_coco_format_test_range_0.00_1.00.json   
│   │   ├── ochuman_coco_format_val_range_0.00_1.00.json   
│   ├── images  
│   │   ├── ####.jpg 
```

配置train.py和test.py中Dataset的文件路径

```python
class Dataset():
    def __init__(self):
        ImageRoot = './data/coco2017/train2017'
        AnnoFile = './data/coco2017/annotations/person_keypoints_train2017_pose2seg.json'
```

```python
if dataset == 'OCHumanVal':
     ImageRoot = './data/OCHuman/images'
     AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json'
elif dataset == 'OCHumanTest':
     ImageRoot = './data/OCHuman/images'
     AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json'
elif dataset == 'cocoVal':
     ImageRoot = './data/coco2017/val2017'
     AnnoFile = './data/coco2017/annotations/person_keypoints_val2017_pose2seg.json'
```

**Yolact:**

修改data/config.py的数据集路径为本地路径

```python
coco2017_dataset = dataset_base.copy({
    'name': 'COCO 2017',
    
    'train_info': './data/coco/annotations/instances_train2017.json',
    'valid_info': './data/coco/annotations/instances_val2017.json',

    'label_map': COCO_LABEL_MAP
})
```

**Detectron:**

配置detectron/config/paths_catalog.py

```python
class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
```



### 2 使用教程

#### 2.1 安装Jittor

```bash
sudo apt install python3.7-dev libomp-dev
sudo python3.7 -m pip install git+https://github.com/Jittor/jittor.git
```



#### 2.2 使用Pose2Seg人体分割模型

下载pretrained模型：[here](https://drive.google.com/file/d/193i8b40MJFxawcJoNLq1sG0vhAeLoVJG/view?usp=sharing)

**Train:**

```bash
python train.py
```

**Test:**

```python
python test.py --weights last.pkl --coco --OCHuman
```



#### 2.3 使用Yolact实时分割模型

Pretrained模型 (来源：https://github.com/dbolya/yolact)：

Here are  YOLACT models (released on April 5th, 2019) along with their FPS on a Titan Xp and mAP on `test-dev`:

| Image Size | Backbone      | FPS  | mAP  | Weights                                                      |                                                              |
| ---------- | ------------- | ---- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 550        | Resnet50-FPN  | 42.5 | 28.2 | [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EUVpxoSXaqNIlssoLKOEoCcB1m0RpzGq_Khp5n1VX3zcUw) |
| 550        | Darknet53-FPN | 40.0 | 28.7 | [yolact_darknet53_54_800000.pth](https://drive.google.com/file/d/1dukLrTzZQEuhzitGkHaGjphlmRJOjVnP/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/ERrao26c8llJn25dIyZPhwMBxUp2GdZTKIMUQA3t0djHLw) |
| 550        | Resnet101-FPN | 33.5 | 29.8 | [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EYRWxBEoKU9DiblrWx2M89MBGFkVVB_drlRd_v5sdT3Hgg) |
| 700        | Resnet101-FPN | 23.6 | 31.2 | [yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/Eagg5RSc5hFEhp7sPtvLNyoBjhlf2feog7t8OQzHKKphjw) |

YOLACT++ models (released on December 16th, 2019):

| Image Size | Backbone      | FPS  | mAP  | Weights                                                      |                                                              |
| ---------- | ------------- | ---- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 550        | Resnet50-FPN  | 33.5 | 34.1 | [yolact_plus_resnet50_54_800000.pth](https://drive.google.com/file/d/1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EcJAtMiEFlhAnVsDf00yWRIBUC4m8iE9NEEiV05XwtEoGw) |
| 550        | Resnet101-FPN | 27.3 | 34.6 | [yolact_plus_base_54_800000.pth](https://drive.google.com/file/d/15id0Qq5eqRbkD-N3ZjDZXdCvRyIaHpFB/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EVQ62sF0SrJPrl_68onyHF8BpG7c05A8PavV4a849sZgEA) |

**Train:**

```bash
# Trains using the base config with a batch size of 8 (the default).
python train.py --config=yolact_base_config

# Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python train.py --config=yolact_base_config --batch_size=5

# Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```

**Test:**

```bash
# Display qualitative results on the specified image.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.png

# Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder
```

#### 2.4 使用Detectron库

**安装：**

```bash
cd detectron.jittor
python setup.py install
```

**配置config （参考configs的样例）**：

```yaml
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"
  WEIGHT: "https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_C4_1x.pth"
  RPN:
    PRE_NMS_TOP_N_TEST: 6000
    POST_NMS_TOP_N_TEST: 1000
  ROI_MASK_HEAD:
    PREDICTOR: "MaskRCNNC4Predictor"
    SHARE_BOX_FEATURE_EXTRACTOR: True
  MASK_ON: True
DATASETS:
  TRAIN: ("coco_2014_train", "coco_2014_valminusminival")
  TEST: ("coco_2014_minival",)
SOLVER:
  BASE_LR: 0.01
  WEIGHT_DECAY: 0.0001
  STEPS: (120000, 160000)
  MAX_ITER: 180000
  IMS_PER_BATCH: 8

```

**使用：**

```python
#coding=utf-8
import requests
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import jittor as jt 
from detectron.config import cfg
from predictor import COCODemo

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

# turn on cuda
jt.flags.use_cuda = 1

# set config
config_file = '../configs/maskrcnn_benchmark/e2e_mask_rcnn_R_50_FPN_1x.yaml'
# update the config options with the config file
cfg.merge_from_file(config_file)
#cfg.MODEL.WEIGHT = "weight/maskrcnn_r50.pth"

#set predictor
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

#load image
pil_image = Image.open('test.jpg').convert("RGB")
image = np.array(pil_image)[:, :, [2, 1, 0]]

# compute predictions
predictions = coco_demo.run_on_opencv_image(image)

# save result
cv2.imwrite('predicton.jpg',predictions)
```

**Train:**

```python
python tools/train_net.py --config-file /path/your/configfile
```

**Test:**

```
python tools/test_net.py --config-file /path/your/configfile
```



### 3 参考

1.  https://github.com/liruilong940607/Pose2Seg
2. https://arxiv.org/abs/1803.10683
3. https://github.com/dbolya/yolact
4. https://arxiv.org/abs/1904.02689
5. https://arxiv.org/abs/1912.06218
6. https://github.com/facebookresearch/maskrcnn-benchmark