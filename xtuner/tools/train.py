# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import logging
import os
import os.path as osp
from functools import partial
from types import FunctionType

from mmengine.config import Config, DictAction
from mmengine.config.lazy import LazyObject
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments

from xtuner.configs import cfgs_name_path
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.model.modules import dispatch_modules
from xtuner.model.utils import LoadWoInit, find_all_linear_names, traverse_dict
from xtuner.registry import BUILDER, MAP_FUNC
from xtuner.tools.utils import auto_dtype_of_deepspeed_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--deepspeed',
        type=str,
        default=None,
        help='the path to the .json file for deepspeed')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    return args


def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if isinstance(value, FunctionType):
                value_str = str(value)
                if value_str not in MAP_FUNC:
                    MAP_FUNC.register_module(module=value, name=value_str)
                cfg_dict[key] = value_str
            else:
                register_function(value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            register_function(value)


def main():
    args = parse_args()

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register FunctionType object in cfg to `MAP_FUNC` Registry and
    # change these FunctionType object to str
    register_function(cfg._cfg_dict)

    if cfg.get('framework', 'mmengine').lower() == 'huggingface':
        # set default training_args
        if cfg.get('training_args', None) is None:
            cfg.training_args = dict(type=TrainingArguments)
        # set work_dir
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.training_args.output_dir = args.work_dir
        elif cfg.training_args.get('output_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.training_args.output_dir = osp.join(
                './work_dirs',
                osp.splitext(osp.basename(args.config))[0])
        # enable deepspeed
        if args.deepspeed:
            if not osp.isfile(args.deepspeed):
                try:
                    args.deepspeed = cfgs_name_path[args.deepspeed]
                except KeyError:
                    raise FileNotFoundError(f'Cannot find {args.deepspeed}')
            cfg.training_args.deepspeed = args.deepspeed
        if cfg.training_args.get('deepspeed'):
            device_map = None
        else:
            # Data Parallel
            device_map = {
                '': int(os.environ.get('LOCAL_RANK', args.local_rank))
            }
        # build training_args
        training_args = BUILDER.build(cfg.training_args)
        # build model
        with LoadWoInit():
            cfg.model.device_map = device_map
            traverse_dict(cfg.model)
            model = BUILDER.build(cfg.model)
        model.config.use_cache = False
        dispatch_modules(model)
        if cfg.get('lora', None):
            lora = BUILDER.build(cfg.lora)
            model = prepare_model_for_kbit_training(model)
            if lora.target_modules is None:
                modules = find_all_linear_names(model)
                lora.target_modules = modules
            model = get_peft_model(model, lora)

        # build dataset
        train_dataset = BUILDER.build(cfg.train_dataset)
        data_collator = partial(default_collate_fn, return_hf_format=True)
        # build trainer
        trainer = cfg.trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator)
        # training
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)
    else:
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        cfg.launcher = args.launcher
        # work_dir is determined in this priority:
        # CLI > segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])

        if args.deepspeed:
            try:
                import deepspeed
            except ImportError:
                raise ImportError(
                    'deepspeed is not installed properly, please check.')
            if digit_version(deepspeed.__version__) < digit_version('0.12.3'):
                raise RuntimeError('Please upgrade your DeepSpeed version '
                                   'by using the command pip install '
                                   '`deepspeed>=0.12.3`')
            optim_wrapper = cfg.optim_wrapper.type
            if optim_wrapper == 'DeepSpeedOptimWrapper':
                print_log(
                    'Deepspeed training is already enabled in your config.',
                    logger='current',
                    level=logging.WARNING)
            else:
                if not osp.isfile(args.deepspeed):
                    try:
                        args.deepspeed = cfgs_name_path[args.deepspeed]
                    except KeyError:
                        raise FileNotFoundError(
                            f'Cannot find {args.deepspeed}')
                with open(args.deepspeed) as f:
                    ds_cfg = json.load(f)

                ds_grad_accum = ds_cfg.get('gradient_accumulation_steps',
                                           'auto')
                mm_grad_accum = cfg.optim_wrapper.get('accumulative_counts', 1)
                if ds_grad_accum != 'auto' and ds_grad_accum != mm_grad_accum:
                    print_log(('Mismatch on gradient_accumulation_steps: '
                               f'MMEngine {mm_grad_accum}, '
                               f'Deepspeed {ds_grad_accum}. '
                               f'Set to {mm_grad_accum}'),
                              logger='current',
                              level=logging.WARNING)
                grad_accum = mm_grad_accum

                ds_train_bs = ds_cfg.get('train_micro_batch_size_per_gpu',
                                         'auto')
                mm_train_bs = cfg.train_dataloader.batch_size
                if ds_train_bs != 'auto' and ds_train_bs != mm_train_bs:
                    print_log(
                        ('Mismatch on train_micro_batch_size_per_gpu: '
                         f'MMEngine {mm_train_bs}, Deepspeed {ds_train_bs}. '
                         f'Set to {mm_train_bs}'),
                        logger='current',
                        level=logging.WARNING)
                train_bs = cfg.train_dataloader.batch_size

                ds_grad_clip = ds_cfg.get('gradient_clipping', 'auto')
                clip_grad = cfg.optim_wrapper.get('clip_grad', None)
                if clip_grad and clip_grad.get('max_norm', None) is not None:
                    mm_max_norm = cfg.optim_wrapper.clip_grad.max_norm
                else:
                    mm_max_norm = 1.0
                if ds_grad_clip != 'auto' and ds_grad_clip != mm_max_norm:
                    print_log(
                        ('Mismatch on gradient_clipping: '
                         f'MMEngine {mm_max_norm}, Deepspeed {ds_grad_clip}. '
                         f'Set to {mm_max_norm}'),
                        logger='current',
                        level=logging.WARNING)
                grad_clip = mm_max_norm
                ds_cfg = auto_dtype_of_deepspeed_config(ds_cfg)
                exclude_frozen_parameters = True if digit_version(
                    deepspeed.__version__) >= digit_version('0.10.1') else None
                strategy = dict(
                    type=LazyObject('xtuner.engine', 'DeepSpeedStrategy'),
                    config=ds_cfg,
                    gradient_accumulation_steps=grad_accum,
                    train_micro_batch_size_per_gpu=train_bs,
                    gradient_clipping=grad_clip,
                    exclude_frozen_parameters=exclude_frozen_parameters)
                cfg.__setitem__('strategy', strategy)
                optim_wrapper = dict(
                    type='DeepSpeedOptimWrapper',
                    optimizer=cfg.optim_wrapper.optimizer)
                cfg.__setitem__('optim_wrapper', optim_wrapper)
                cfg.runner_type = 'FlexibleRunner'

        # resume is determined in this priority: resume from > auto_resume
        if args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # start training
        runner.train()


if __name__ == '__main__':
    main()
