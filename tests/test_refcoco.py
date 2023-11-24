# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from unittest import TestCase


from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
import copy
from xtuner.dataset import RefCOCOTrainDataset, InvRefCOCOTrainDataset
from xtuner.dataset.refcoco import Blip2ImageTrainProcessor
from xtuner.dataset.map_fns import refcoco_map_fn
from xtuner.registry import BUILDER
import torch


import logging

PATH = "xtuner/configs/llava/llava_vicuna_7b_v15_clip_vit_large_p14_e1_gpu8_pretrain.py"


class TestRef(TestCase):

    def test_ref(self):
        config_path = PATH
        dataset_config = Config.fromfile(config_path)['llava_dataset']

        refcoco_dataset_config = copy.copy(dataset_config)
        refcoco_dataset_config['type'] = RefCOCOTrainDataset
        refcoco_dataset_config['data_path'] = 'data/refcoco/refcoco_annotations'
        refcoco_dataset_config['image_folder'] = 'data/refcoco/train2014'
        refcoco_dataset_config['dataset_map_fn'] = refcoco_map_fn
        refcoco_dataset_config['processor'] = dict(
            type=Blip2ImageTrainProcessor
        )
        refcoco_set = BUILDER.build(refcoco_dataset_config)
        item = refcoco_set[0]
        self._print(item)

    def test_inv_ref(self):
        config_path = PATH
        dataset_config = Config.fromfile(config_path)['llava_dataset']

        refcoco_dataset_config = copy.copy(dataset_config)
        refcoco_dataset_config['type'] = InvRefCOCOTrainDataset
        refcoco_dataset_config['data_path'] = 'data/refcoco/refcoco_annotations'
        refcoco_dataset_config['image_folder'] = 'data/refcoco/train2014'
        refcoco_dataset_config['dataset_map_fn'] = refcoco_map_fn

        refcoco_set = BUILDER.build(refcoco_dataset_config)
        item = refcoco_set[0]
        self._print(item)

    def test_llava(self):
        config_path = PATH
        dataset_config = Config.fromfile(config_path)['llava_dataset']

        dataset = BUILDER.build(dataset_config)
        item = dataset[0]
        self._print(item)

    def _print(self, item):
        for key in item:
            value = item[key]
            if isinstance(value, torch.Tensor):
                print(f"{key}\n{value.shape}")
            elif isinstance(value, list):
                print(f"{key}\n{value}\n{len(value)}")
            else:
                print(f"{key}\n{value}")
            print()
