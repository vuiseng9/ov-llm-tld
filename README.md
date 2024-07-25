# 24h1-sparse-quantized-llm-ov

## 2024 July 24 (Branch: 240724-llama2-50)
```bash
pip install deepsparse-nightly[llm] # deepsparse-nightly==1.8.0.20240502

# reproduce NM's
./deepsparse_reproduce_llama2.bash
# official results
# https://sparsezoo.neuralmagic.com/models/llama2-7b-gsm8k_llama2_pretrain-pruned50?comparison=llama2-7b-gsm8k_llama2_pretrain-base&hardware=deepsparse-m7i.4xlarge
# https://sparsezoo.neuralmagic.com/models/llama2-7b-gsm8k_llama2_pretrain-pruned50_quantized?comparison=llama2-7b-gsm8k_llama2_pretrain-base&hardware=deepsparse-m7i.4xlarge
```

## setup

Install pytorch and:

```bash
pip install tabulate transformers==4.35  optimum-intel[openvino]==1.13.0 nncf==2.7.0
pip install deepsparse-nightly[llm]==1.6.0.20231120
pip install openvino==2023.3.0
```

## benchmark

- Reproduce NM paper: `deepsparse_reproduce.bash`
- Export IR models: `export_ir.bash` & `export_ir_w_mask.bash`(sparse models)
- OV benchmarkapp: see the files: `benchmarkapp_*.bash`, or see `run_ov_benchmark_app.bash` (more complicated; codes are not well organized).
