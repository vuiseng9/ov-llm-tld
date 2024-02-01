for sparsity in "70"; do
    echo ">>>> Sparsity=$sparsity"
    run_name=w8a8-sparse$sparsity
    folder=./models/neuralmagic/mpt-7b-gsm8k-pt/$run_name
    mkdir -p $folder
    python export_ir_w_mask.py --quant_mode W8A8 \
        --ref_sparse_onnx neuralmagic/mpt-7b-gsm8k-pruned$sparsity-quant-ds \
        --run_name $run_name \
        --force_run 2>&1 | tee $folder/log.log
done
