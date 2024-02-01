import copy
import json
import os
import typing
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import transformers
from openvino.runtime import Core
from optimum.intel.openvino import OVConfig, OVQuantizer
from transformers import HfArgumentParser


@dataclass
class Args:
    model_id: str
    save_dir: str
    sparsity: float


W8A8_QUANTIZATION_CONFIG = {
    "algorithm": "quantization",
    "preset": "performance",  # sym weight, sym activation
    "overflow_fix": "disable",
    "initializer": {
        "range": {"num_init_samples": 16, "type": "min_max"},
        "batchnorm_adaptation": {"num_bn_adaptation_samples": 0},
    },
    "ignored_scopes": [
        # "{re}.*Embedding.*",
        # "{re}.*add___.*",
        # layer norm. Models other than llama/opt might need to set another name.
        "{re}.*norm.*",
        # "{re}.*matmul_1",
        # "{re}.*__truediv__.*",
    ],
}

ie = Core()


def _model_has_layer_norm(model):
    for name, _ in model.named_parameters():
        if 'layernorm' in name.lower().replace('_', ''):
            return True
    return False


@torch.no_grad()
def _sparsify_param(param: torch.nn.Parameter, sparsity: float):
    k = min(param.numel(), int(param.numel() * sparsity) + 1)
    abs_data = param.data.abs()
    threshold = float(torch.kthvalue(abs_data.view(-1), k).values)
    param.data[abs_data <= threshold] = 0.


def _get_named_linear_layers(model: transformers.PreTrainedModel):
    if model.__class__.__name__ == 'OPTForCausalLM':
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.startswith('model.decoder.layers.'):
                yield name, module
    elif model.__class__.__name__ == 'LlamaForCausalLM':
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.startswith('model.layers.'):
                yield name, module
    else:
        raise NotImplementedError(model.__class__.__name__)


def sparsify(model: transformers.PreTrainedModel, sparsity: float):
    for name, module in _get_named_linear_layers(model):
        module: torch.nn.Linear
        _sparsify_param(module.weight, sparsity=sparsity)
        if module.bias is not None:
            _sparsify_param(module.bias, sparsity=sparsity)
        assert (module.weight == 0).float().mean() >= sparsity * 0.95
    return model


def quantize_and_save(model: transformers.PreTrainedModel, tokenizer, save_dir):
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte") \
            and type(model.transformer.wte) != torch.nn.Embedding:
        print(model.config_class)
        from nncf.torch import register_module
        register_module(ignored_algorithms=[])(type(model.transformer.wte))

    def preprocess_fn(examples, tokenizer):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=32)

    compression_config = copy.deepcopy(W8A8_QUANTIZATION_CONFIG)
    if not _model_has_layer_norm(model):
        print(
            'Deleting ignored_scope of layer_norm since the model does not have layernorm')
        compression_config['ignored_scopes'].remove("{re}.*layer_norm_.*")

    quantizer = OVQuantizer.from_pretrained(model)
    calibration_dataset = quantizer.get_calibration_dataset(
        "wikitext",
        dataset_config_name="wikitext-2-v1",
        num_samples=20,
        dataset_split='train',
        preprocess_function=partial(preprocess_fn, tokenizer=tokenizer)
    )
    ov_config = OVConfig(compression_config)
    ov_config.log_dir = str(save_dir)
    quantizer.quantize(
        calibration_dataset=calibration_dataset,
        save_directory=save_dir,
        quantization_config=ov_config,
    )
    tokenizer.save_pretrained(save_dir)


def check_ov_w8a8_ir_sparsity(ov_xml_path: str):
    result = {}
    ovmodel = ie.read_model(model=ov_xml_path)
    for op in ovmodel.get_ordered_ops():
        if "constant" in str(op.get_type_info()).lower() and 'int8_t' in str(op.get_element_type()):
            sparsity = (op.get_data() == 0).astype(float).mean()
            result[op.get_friendly_name()] = sparsity
    with open(Path(ov_xml_path).parent / 'ov_sparsity_stats.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print('OV model quantized linear layer sparsity:\n',
          json.dumps(result, indent=2))


def main(model_id: str, save_dir: str, sparsity: float):
    xml_name = 'openvino_model.xml'
    if Path(save_dir, xml_name).exists():
        return
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    model = sparsify(model, sparsity=sparsity)
    quantize_and_save(model, tokenizer, save_dir)
    check_ov_w8a8_ir_sparsity(Path(save_dir, xml_name).as_posix())
    print(f'Quantized sparse ov model saved at {save_dir}.')


if __name__ == "__main__":
    # main('facebook/opt-350m', './logs/opt-350m-w8a8-unstructured50/', sparsity=0.5)
    # main('facebook/opt-350m', './logs/opt-350m-w8a8-unstructured90/', sparsity=0.9)
    # main('facebook/opt-6.7b', './logs/opt-6.7b-w8a8-unstructured50/', sparsity=0.5)
    # main('facebook/opt-6.7b', './logs/opt-6.7b-w8a8-unstructured90/', sparsity=0.9)
    # main('meta-llama/Llama-2-7b-hf', './logs/llama-2-7b-w8a8-unstructured50/', sparsity=0.5)
    # main('meta-llama/Llama-2-13b-hf', './logs/llama-2-13b-w8a8-unstructured50/', sparsity=0.5)
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    args = typing.cast(Args, args)
    main(args.model_id, args.save_dir, args.sparsity)
