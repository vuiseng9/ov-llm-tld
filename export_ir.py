import datetime
import gc
import json
import logging
import multiprocessing
import os
import subprocess
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import nncf
import tabulate  # required by df.to_markdown
import torch
import transformers
import yaml
from optimum.exporters.openvino.__main__ import TasksManager
from optimum.intel import OVConfig, OVModelForCausalLM, OVQuantizer
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, HfArgumentParser)

import show_ir_weights

nncf.set_log_level(logging.ERROR)

QUANT_MODES = ['original', 'W8', 'W4asym', 'W8A8',
               #    'Wnf4', 'W4sym',
               ]


@dataclass
class Args:
    model_id: str = 'neuralmagic/mpt-7b-gsm8k-pt'
    run_name: str = 'fp32'
    quant_mode: str = field(default='original', metadata={
                            "choices": QUANT_MODES})
    force_run: bool = False

    @property
    def save_dir(self):
        folder = Path('./models/', self.model_id, self.run_name)
        folder.mkdir(parents=True, exist_ok=True)
        return folder.as_posix()


@contextmanager
def patch_ovmodel_creation(torch_model: transformers.PreTrainedModel):
    """
    When doing `OVModelForCausalLM.from_pretrained(model_id)`, internally it will use this torch model,
    ignoring the `model_id` argument.
    """

    class TasksManagerPatched(TasksManager):
        def get_model_from_task(self, *args, **kwargs):
            print('Using patched torch_model...')
            return torch_model

    with patch('optimum.exporters.openvino.__main__.TasksManager', TasksManagerPatched):
        yield


@contextmanager
def patch_torch_onnx_export(do_constant_folding: bool):
    ori_func = torch.onnx.export

    def patched_func(*args, **kwargs):
        kwargs.pop('do_constant_folding')
        print(f'** Patching do_constant_folding to {do_constant_folding}...')
        return ori_func(*args, do_constant_folding=do_constant_folding, **kwargs)

    with patch('torch.onnx.export', patched_func) as patcher:
        yield patcher


def dump_json(obj, file_path):
    Path(file_path).parent.mkdir(exist_ok=True, parents=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def create_ov_model(model_id: str, torch_model: transformers.PreTrainedModel):
    with patch_ovmodel_creation(torch_model):
        with patch_torch_onnx_export(do_constant_folding=False):
            ov_model = OVModelForCausalLM.from_pretrained(
                model_id=model_id,
                config=torch_model.config,
                export=True,
                compile=False,
                ov_config={
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_PRECISION_HINT": "f32",  # nncf.compress_weights does not support bf16
                }
            )
            return ov_model


def w8a8_quantize(args: Args, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
    w8a8_compression = {
        "algorithm": "quantization",
        "preset": "performance",
        "overflow_fix": "disable",
        "initializer": {
            "range": {"num_init_samples": 4, "type": "mean_min_max"},
            "batchnorm_adaptation": {"num_bn_adaptation_samples": 0},
        },
        "scope_overrides": {"activations": {"{re}.*matmul_0": {"mode": "symmetric"}}},
        "ignored_scopes": [
            # "{re}.*Embedding.*",
            # "{re}.*add___.*",
            "{re}.*layer_norm_.*",
            # "{re}.*matmul_1",
            # "{re}.*__truediv__.*",
        ],
    }
    ov_config = OVConfig(compression=w8a8_compression)
    ov_config.log_dir = args.save_dir
    tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "transformer") and hasattr(model.transformer, "wte") and type(model.transformer.wte) != torch.nn.Embedding:
        print(model.config_class)
        from nncf.torch import register_module
        register_module(ignored_algorithms=[])(type(model.transformer.wte))

    def preprocess_fn(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=8)

    quantizer = OVQuantizer.from_pretrained(model)
    calibration_dataset = quantizer.get_calibration_dataset(
        "yujiepan/wikitext-tiny",
        num_samples=8,
        dataset_split='train',
        preprocess_function=preprocess_fn,
    )
    quantizer.quantize(calibration_dataset=calibration_dataset, save_directory=args.save_dir,
                       quantization_config=ov_config, weights_only=False)


def main(args: Args):
    xml_path = Path(args.save_dir, 'openvino_model.xml')
    if xml_path.exists() and not args.force_run:
        return

    config = AutoConfig.from_pretrained(args.model_id)
    config.use_cache = True
    config.tie_word_embeddings = False
    config.tie_weights = False
    torch_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, config=config)
    # torch_model = AutoModelForCausalLM.from_config(config)
    for name, param in torch_model.named_parameters():
        print(name, param.shape)
    for name, _ in torch_model.named_modules():
        print(name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(args.save_dir)

    if args.quant_mode == 'W8A8':
        w8a8_quantize(args, torch_model, tokenizer)
    else:
        ov_model = create_ov_model(args.model_id, torch_model)
        # ov_model = ov_model.half()
        if args.quant_mode == 'original':
            pass
        elif args.quant_mode == 'W8':
            ov_model.model = nncf.compress_weights(ov_model.model,
                                                   mode=nncf.CompressWeightsMode.INT8,
                                                   group_size=-1)
        elif args.quant_mode == 'W4asym':
            ov_model.model = nncf.compress_weights(ov_model.model,
                                                   mode=nncf.CompressWeightsMode.INT4_ASYM,
                                                   group_size=-1)
        elif args.quant_mode == 'W4sym':
            ov_model.model = nncf.compress_weights(ov_model.model,
                                                   mode=nncf.CompressWeightsMode.INT4_SYM,
                                                   group_size=-1)
        elif args.quant_mode == 'Wnf4':
            ov_model.model = nncf.compress_weights(ov_model.model,
                                                   mode=nncf.CompressWeightsMode.NF4,
                                                   group_size=-1)
        ov_model.save_pretrained(args.save_dir)

    dump_json(args.__dict__, Path(args.save_dir, 'args.json'))
    df = show_ir_weights.main(show_ir_weights.Args(
        xml_path=xml_path.as_posix(),
    ))
    df.to_markdown(Path(args.save_dir, 'ov_weights_type.md'))


if __name__ == '__main__':
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    print(args)
    main(args)
