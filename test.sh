CUDA_VISIBLE_DEVICES=0
python testImages.py \
--mode val \
--path lsp_dataset/lsp_dataset \
--config config.default_config \
--use_gpu \
--train_split 0.5 \
--batch_size 1 \
--modelName LSP-keepdimension/LSP12000_VERSION02/model_229_1300.pt




# pre-trained generator with keepdimension 

