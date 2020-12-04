# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from detectron.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import jittor as jt
from detectron.config import cfg
from detectron.data import make_data_loader
from detectron.solver import make_lr_scheduler
from detectron.solver import make_optimizer
from detectron.engine.inference import inference
from detectron.engine.trainer import do_train
from detectron.modeling.detector import build_detection_model
from detectron.utils.checkpoint import DetectronCheckpointer
from detectron.utils.collect_env import collect_env_info
from detectron.utils.imports import import_file
from detectron.utils.logger import setup_logger
from detectron.utils.miscellaneous import mkdir, save_config


def train(cfg, local_rank, distributed):
    model = build_detection_model(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)


    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        checkpoint_period,
        test_period,
        arguments,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/home/lxl/jittor/detectron.jittor/configs/maskrcnn_benchmark/e2e_faster_rcnn_R_50_C4_1x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("detectron", output_dir)
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    # logger.info("Loaded configuration file {}".format(args.config_file))

    model = train(cfg, args.local_rank, False)

    if not args.skip_test:
        run_test(cfg, model, False)


if __name__ == "__main__":
    main()
