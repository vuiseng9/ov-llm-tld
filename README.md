# 24h1-sparse-quantized-llm-ov

## setup

Install pytorch and:

```bash
pip install torch==2.3.1 https://download.pytorch.org/whl/cpu
pip install tabulate transformers==4.42.4  optimum-intel[openvino]==1.18.1 nncf==2.11.0
pip install datasets==2.20.0 accelerate==0.32.1
pip install openvino==2024.2.0
# pip install deepsparse-nightly[llm]==1.6.0.20231120
```

## benchmark

- Reproduce NM paper: `deepsparse_reproduce.bash`
- Export IR models: `export_ir.bash` & `export_ir_w_mask.bash`(sparse models)
- OV benchmarkapp: see the files: `benchmarkapp_*.bash`, or see `run_ov_benchmark_app.bash` (more complicated; codes are not well organized).
