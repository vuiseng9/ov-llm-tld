benchmark_app -m ./models/neuralmagic/mpt-7b-gsm8k-pt/fp32/openvino_model.xml -data_shape input_ids[1,1],attention_mask[1,512],past_key_values.0.value[1,32,511,128],past_key_values.0.key[1,32,511,128],past_key_values.1.value[1,32,511,128],past_key_values.1.key[1,32,511,128],past_key_values.2.value[1,32,511,128],past_key_values.2.key[1,32,511,128],past_key_values.3.value[1,32,511,128],past_key_values.3.key[1,32,511,128],past_key_values.4.value[1,32,511,128],past_key_values.4.key[1,32,511,128],past_key_values.5.value[1,32,511,128],past_key_values.5.key[1,32,511,128],past_key_values.6.value[1,32,511,128],past_key_values.6.key[1,32,511,128],past_key_values.7.value[1,32,511,128],past_key_values.7.key[1,32,511,128],past_key_values.8.value[1,32,511,128],past_key_values.8.key[1,32,511,128],past_key_values.9.value[1,32,511,128],past_key_values.9.key[1,32,511,128],past_key_values.10.value[1,32,511,128],past_key_values.10.key[1,32,511,128],past_key_values.11.value[1,32,511,128],past_key_values.11.key[1,32,511,128],past_key_values.12.value[1,32,511,128],past_key_values.12.key[1,32,511,128],past_key_values.13.value[1,32,511,128],past_key_values.13.key[1,32,511,128],past_key_values.14.value[1,32,511,128],past_key_values.14.key[1,32,511,128],past_key_values.15.value[1,32,511,128],past_key_values.15.key[1,32,511,128],past_key_values.16.value[1,32,511,128],past_key_values.16.key[1,32,511,128],past_key_values.17.value[1,32,511,128],past_key_values.17.key[1,32,511,128],past_key_values.18.value[1,32,511,128],past_key_values.18.key[1,32,511,128],past_key_values.19.value[1,32,511,128],past_key_values.19.key[1,32,511,128],past_key_values.20.value[1,32,511,128],past_key_values.20.key[1,32,511,128],past_key_values.21.value[1,32,511,128],past_key_values.21.key[1,32,511,128],past_key_values.22.value[1,32,511,128],past_key_values.22.key[1,32,511,128],past_key_values.23.value[1,32,511,128],past_key_values.23.key[1,32,511,128],past_key_values.24.value[1,32,511,128],past_key_values.24.key[1,32,511,128],past_key_values.25.value[1,32,511,128],past_key_values.25.key[1,32,511,128],past_key_values.26.value[1,32,511,128],past_key_values.26.key[1,32,511,128],past_key_values.27.value[1,32,511,128],past_key_values.27.key[1,32,511,128],past_key_values.28.value[1,32,511,128],past_key_values.28.key[1,32,511,128],past_key_values.29.value[1,32,511,128],past_key_values.29.key[1,32,511,128],past_key_values.30.value[1,32,511,128],past_key_values.30.key[1,32,511,128],past_key_values.31.value[1,32,511,128],past_key_values.31.key[1,32,511,128] -t 120 -hint latency  -infer_precision f32