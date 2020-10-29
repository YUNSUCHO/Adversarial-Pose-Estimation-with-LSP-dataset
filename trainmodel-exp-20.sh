
python trainmodel-exp-20.py \
--path lsp_dataset/lsp_dataset \
--modelName LSP-keepdimension/LSP12000_VERSION02_after230epochs \
--config config.default_config \
--batch_size 8 \
--use_gpu \
--gpu_device 0 \
--lr .00025 \
--print_every 1 \
--train_split 0.9167 \
--loss mse \
--optimizer_type Adam \
--epochs 30 \
--dataset  'lsp' 


