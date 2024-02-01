# mpt-7b-gsm8k_mpt_pretrain-pruned60 \
# mpt-7b-gsm8k_mpt_pretrain-pruned60_quantized \

models="mpt-7b-gsm8k_mpt_pretrain-base \
    mpt-7b-gsm8k_mpt_pretrain-base_quantized \
    mpt-7b-gsm8k_mpt_pretrain-pruned50 \
    mpt-7b-gsm8k_mpt_pretrain-pruned50_quantized \
    mpt-7b-gsm8k_mpt_pretrain-pruned70 \
    mpt-7b-gsm8k_mpt_pretrain-pruned70_quantized"

mkdir ds.benchmark/
for model in $models; do
    cmd="deepsparse.benchmark zoo:$model -t 120"
    echo *****$cmd***** | tee -a ds.benchmark/$model.log
    eval $cmd 2>&1 | tee -a ds.benchmark/$model.log
done
