
python trainmodel-adversarial-mode-exp24.py \
--path lsp_dataset/lsp_dataset \
--modelName pre-train-discriminator-with-LSP-keepdimension \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--gpu_device 0 \
--lr .00025 \
--print_every 1 \
--train_split 0.9167 \
--loss mse \
--optimizer_type Adam \
--epochs 230 \
--dataset  'lsp' 


