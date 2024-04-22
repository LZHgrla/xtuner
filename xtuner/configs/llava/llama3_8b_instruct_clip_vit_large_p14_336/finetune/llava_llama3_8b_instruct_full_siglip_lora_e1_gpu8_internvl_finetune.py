# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          SiglipVisionModel, SiglipImageProcessor)

from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import mm_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.model import LLaVAModel
from peft import LoraConfig
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.dataset.evaluation import MMEDataset, MultipleChoiceDataset, POPEDataset, \
    HallusionDataset, TextVQADataset, GQADataset
from xtuner.dataset import ConcatDataset
from xtuner.engine.runner import TrainLoop, ValLoop, TestLoop

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = '/mnt/petrelfs/share_data/fanqi/meta-llama/Meta-Llama-3-8B-Instruct'
visual_encoder_name_or_path = 'google/siglip-so400m-patch14-384'
# Specify the pretrained pth
pretrained_pth = './work_dirs/llava_llama3_8b_instruct_siglip_e1_gpu8_sharegpt4v_pretrain/iter_9742.pth'  # noqa: E501
# Data
data_root = './data/internvl_sft/'

sharegpt4v_caption_data_path = data_root + 'sharegpt4v_instruct_gpt4-vision_cap100k.jsonl'
sharegpt4v_caption_image_folder = data_root + 'data'

llava_data_path = data_root + 'llava_instruct_150k_zh.jsonl'
llava_image_folder = data_root + 'data/coco'

sharegpt4v_data_path = data_root + 'sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl'
sharegpt4v_image_folder = data_root + 'data'

dvqa_data_path = data_root + 'dvqa_train_200k.jsonl'
dvqa_image_folder = data_root + 'data/dvqa'

chartqa_data_path = data_root + 'chartqa_train_18k.jsonl'
chartqa_image_folder = data_root + 'data/chartqa'

ai2d_data_path = data_root + 'ai2d_train_12k.jsonl'
ai2d_image_folder = data_root + 'data/ai2d'

docvqa_data_path = data_root + 'docvqa_train_10k.jsonl'
docvqa_image_folder = data_root + 'data/docvqa'

geoqa_data_path = data_root + 'geoqa+.jsonl'
geoqa_image_folder = data_root + 'data/geoqa+'

synthdog_data_path = data_root + 'synthdog_en.jsonl'
synthdog_image_folder = data_root + 'data/synthdog-en'

prompt_template = PROMPT_TEMPLATE.llama3_chat
max_length = int(4096 - (384 // 14)**2)

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 4
dataloader_num_workers = 0
max_epochs = 1
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 5000
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)


#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=SiglipImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

model = dict(
    type=LLaVAModel,
    tokenizer=tokenizer,
    template=prompt_template,
    image_processor=image_processor,
    freeze_llm=False,
    freeze_visual_encoder=True,
    pretrained_pth=pretrained_pth,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True),
    visual_encoder=dict(
        type=SiglipVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path),
    visual_encoder_lora=dict(
        type=LoraConfig, r=64, lora_alpha=16, lora_dropout=0.05, bias='none'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
sharegpt4v_caption_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder='./data/llama3_8b_instruct_finetune_sharegpt4v_caption',
    data_path=sharegpt4v_caption_data_path,
    image_folder=sharegpt4v_caption_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

llava_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder='./data/llama3_8b_instruct_finetune_llava_zh',
    data_path=llava_data_path,
    image_folder=llava_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

sharegpt4v_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder='./data/llama3_8b_instruct_finetune_sharegpt4v',
    data_path=sharegpt4v_data_path,
    image_folder=sharegpt4v_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)


dvqa_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder='./data/llama3_8b_instruct_finetune_dvqa',
    data_path=dvqa_data_path,
    image_folder=dvqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

chartqa_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder='./data/llama3_8b_instruct_finetune_chartqa',
    data_path=chartqa_data_path,
    image_folder=chartqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

ai2d_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder='./data/llama3_8b_instruct_finetune_ai2d',
    data_path=ai2d_data_path,
    image_folder=ai2d_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

docvqa_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder='./data/llama3_8b_instruct_finetune_docvqa',
    data_path=docvqa_data_path,
    image_folder=docvqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

geoqa_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder='./data/llama3_8b_instruct_finetune_geoqa',
    data_path=geoqa_data_path,
    image_folder=geoqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

synthdog_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder='./data/llama3_8b_instruct_finetune_synthdog',
    data_path=synthdog_data_path,
    image_folder=synthdog_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[sharegpt4v_caption_dataset, llava_dataset, sharegpt4v_dataset,
              dvqa_dataset, chartqa_dataset, ai2d_dataset, docvqa_dataset,
              geoqa_dataset, synthdog_dataset])

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=mm_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, val_interval=save_steps)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    # dict(
    #     type=EvaluateChatHook,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     every_n_iters=evaluation_freq,
    #     evaluation_inputs=evaluation_inputs,
    #     evaluation_images=evaluation_images,
    #     system=SYSTEM,
    #     prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

# ==================== val and test cfg =======================
val_dataset = [
    dict(
        type=MMEDataset,
        data_file='/mnt/petrelfs/linzhihao/LMUData/MME.tsv',
        image_folder='/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
]

test_dataset = [
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/linzhihao/LMUData/MMBench_DEV_EN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/linzhihao/LMUData/MMBench_TEST_EN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/linzhihao/LMUData/SEEDBench_IMG.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/linzhihao/LMUData/ScienceQA_TEST.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/linzhihao/LMUData/MMMU_DEV_VAL.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/linzhihao/LMUData/AI2D_TEST.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=TextVQADataset,
        data_file='/mnt/petrelfs/share_data/huanghaian/orig_llava_eval/textvqa/llava_textvqa_val_v051_ocr.jsonl',
        ann_file='/mnt/petrelfs/share_data/huanghaian/text_vqa/TextVQA_0.5.1_val.json',
        image_folder='/mnt/petrelfs/share_data/huanghaian/text_vqa/train_images',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MMEDataset,
        data_file='/mnt/petrelfs/linzhihao/LMUData/MME.tsv',
        image_folder='/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        # for_llava_prompt=True, # 开了后，perception 会掉
        pad_image_to_square=True),
    dict(
        type=HallusionDataset,
        data_file='/mnt/petrelfs/linzhihao/LMUData/HallusionBench.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=POPEDataset,
        data_file=[
            '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_adversarial.json',
            '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_popular.json',
            '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_random.json'
        ],
        coco_val_path='/mnt/petrelfs/share_data/linzhihao/dataset/coco/val2014/',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=GQADataset,
        data_file='/mnt/petrelfs/share_data/zhaoxiangyu/gqa_llava_eval/llava_gqa_testdev_balanced.jsonl',
        ann_file='/mnt/petrelfs/share_data/zhaoxiangyu/gqa_llava_eval/testdev_balanced_questions.json',
        image_folder='/mnt/petrelfs/share_data/basemodel/dataset/multimodality/gqa/images',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/share_data/zhaoxiangyu/datasets--Lin-Chen--MMStar/snapshots/mmstar/MMStar.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
]

# TODO: We are not currently using val_evaluator
# Don't support num_workers > 0
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=val_dataset),
    collate_fn=dict(type=mm_collate_fn, extra_collate_keys=['img_id']))
val_evaluator = dict()
val_cfg = dict(type=ValLoop)

# TODO: We are not currently using test_evaluator
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=test_dataset),
    collate_fn=dict(type=mm_collate_fn, extra_collate_keys=['img_id'])
)

test_evaluator = val_evaluator
test_cfg = dict(type=TestLoop, select_metric='first')
