import numpy as np
from onnx import numpy_helper
import onnx
from huggingface_hub import hf_hub_download
import pandas as pd
from pathlib import Path


def download(model_id='neuralmagic/mpt-7b-gsm8k-pruned70-quant-ds'):
    hf_hub_download(repo_id=model_id, filename="model.data")
    return hf_hub_download(repo_id=model_id, filename="model.onnx")


def ref_mask_from_onnx(model_id='neuralmagic/mpt-7b-gsm8k-pruned70-quant-ds'):
    print('Downloading...', model_id)
    onnx_path = download(model_id)
    print('Loading onnx...')
    onnx_model = onnx.load(onnx_path)
    onnx_meta = {}
    for initializer in onnx_model.graph.initializer:
        W = numpy_helper.to_array(initializer)
        name = initializer.name
        if np.prod(W.shape) > 10 and len(W.shape) == 2:
            is_sparse = (W == 128).astype(bool)
            sparsity = is_sparse.astype(np.float32).mean().item()
            onnx_meta[name] = dict(
                name=name,
                min=W.min(),
                max=W.max(),
                sparsity=sparsity,
                is_sparse=is_sparse,
            )
            print(name, sparsity, W.shape)

    return onnx_meta


if __name__ == '__main__':
    ref_mask_from_onnx()
