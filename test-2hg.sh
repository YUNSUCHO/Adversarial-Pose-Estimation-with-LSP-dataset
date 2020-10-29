CUDA_VISIBLE_DEVICES=0
python testImages.py \
--mode val \
--path lspet_dataset \
--config config.default_config \
--use_gpu \
--modelName train-model-baseline-2stackhg-0/model_39_4000.pt




#39 epoch : 
#35 epoch : 72.57
#30 epoch : 73.6
#25 rpoch : 71.9
#20 epoch : 71.4
#15 epoch : 71.7
#10 epoch : 71.2
#5 epoch : 69.3