ov_version=$(pip list | grep openvino | head -n 1 | awk '{print $2}')
echo "Using OV=$ov_version"

for infer_precision in f32 bf16; do
    for use_data_shape in True; do
        for inference_num_threads in 1 8; do
            for model_type in w8a8-sparse70 w8a8-sparse50 fp32 w8a8; do
                # for model_type in w8a8-sparse70; do
                folder=./ov.benchmarkapp/ov$ov_version/infer-precision_$infer_precision,use-data-shape_$use_data_shape,time120,ctx511,inference-num-threads_$inference_num_threads/$model_type
                mkdir -p $folder
                echo ">>>>>>>$folder"
                pip list >$folder/pip.txt
                python -u run_ov_benchmark_app.py \
                    --xml_path ./models/neuralmagic/mpt-7b-gsm8k-pt/$model_type/openvino_model.xml \
                    --save_path $folder \
                    --add_bind False \
                    --infer_precision $infer_precision \
                    --use_data_shape $use_data_shape \
                    --inference_num_threads $inference_num_threads \
                    --time 120 \
                    --ctx_len 511 2>&1 | tee $folder/log.log
                echo ""
            done
        done
    done
done
