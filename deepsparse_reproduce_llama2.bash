models="llama2-7b-gsm8k_llama2_pretrain-base \
    llama2-7b-gsm8k_llama2_pretrain-base_quantized \
    llama2-7b-gsm8k_llama2_pretrain-pruned50_quantized"

mkdir ds.benchmark/
for model in $models; do
    cmd="deepsparse.benchmark zoo:$model -t 120"
    echo *****$cmd***** | tee -a ds.benchmark/$model.log
    eval $cmd 2>&1 | tee -a ds.benchmark/$model.log
done
