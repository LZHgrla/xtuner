# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .concat_dataset import ConcatDataset
from .huggingface import process_hf_dataset
from .intern_repo import (build_packed_dataset,
                          load_intern_repo_tokenized_dataset,
                          load_intern_repo_untokenized_dataset)

from .llava import LLaVADataset, AnyResLLaVADataset, InternVL_V1_5_LLaVADataset
from .json_dataset import load_json_file
from .modelscope import process_ms_dataset
from .moss_sft import MOSSSFTDataset
from .refcoco_json import (InvRefCOCOJsonDataset, RefCOCOJsonDataset,
                           RefCOCOJsonEvalDataset)
from .utils import decode_base64_to_image, expand2square, load_image, internvl_1_5_encode_fn
from .llava_proxy_eval_dataset import LLaVAProxyEvalDataset
from .anyres_llava_proxy_eval_dataset import AnyResLLaVAProxyEvalDataset
from .mini_gemini_dataset import MiniGeminiDataset
from .mini_gemini_proxy_eval_dataset import MiniGeminiProxyEvalDataset
from .internvl_v1_5_llava_proxy_eval_dataset import InternVL_v1_5_LLaVAProxyEvalDataset
from .llava_proxy_eval_dataset1 import LLaVAProxyEvalDataset1
# ignore FutureWarning in hf datasets
warnings.simplefilter(action='ignore', category=FutureWarning)

__all__ = [
    'process_hf_dataset', 'ConcatDataset', 'MOSSSFTDataset',
    'process_ms_dataset', 'LLaVADataset', 'expand2square',
    'decode_base64_to_image', 'load_image', 'process_ms_dataset',
    'load_intern_repo_tokenized_dataset',
    'load_intern_repo_untokenized_dataset',
    'build_packed_dataset',
    'RefCOCOJsonDataset',
    'RefCOCOJsonEvalDataset',
    'InvRefCOCOJsonDataset',
    'AnyResLLaVADataset',
    'load_json_file',
    'LLaVAProxyEvalDataset',
    'AnyResLLaVAProxyEvalDataset',
    'MiniGeminiDataset',
    'MiniGeminiProxyEvalDataset',
    'InternVL_V1_5_LLaVADataset',
    'InternVL_v1_5_LLaVAProxyEvalDataset',
    'internvl_1_5_encode_fn'
]
