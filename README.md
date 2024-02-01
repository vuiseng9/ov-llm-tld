# 24h1-sparse-quantized-llm-ov

## setup

Install pytorch and:

```bash
pip install tabulate transformers  optimum-intel[openvino]==1.13.0 nncf==2.7.0
pip install deepsparse-nightly[llm]==1.6.0.20231120
pip install openvino==2023.3.0
```

## benchmark

- Reproduce NM paper: `ds_reproduce.bash`
- Export IR models: `export_ir.bash` & `export_ir_w_mask.bash`(sparse models)
- OV benchmarkapp: see the files: `benchmarkapp_*.bash`, or see `run_ov_benchmark_app.bash` (more complicated; codes are not well organized).
