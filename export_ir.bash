mkdir -p ./models/neuralmagic/mpt-7b-gsm8k-pt/fp32/
mkdir -p ./models/neuralmagic/mpt-7b-gsm8k-pt/w8a8/
python export_ir.py --quant_mode original --run_name fp32 --force_run 2>&1 | tee -a ./models/neuralmagic/mpt-7b-gsm8k-pt/fp32/log.log
python export_ir.py --quant_mode W8A8 --run_name w8a8 --force_run 2>&1 | tee -a ./models/neuralmagic/mpt-7b-gsm8k-pt/w8a8/log.log
