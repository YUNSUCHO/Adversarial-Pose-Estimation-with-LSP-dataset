CUDA_VISIBLE_DEVICES=1
python testImages.py \
--mode val \
--path lsp_dataset/lsp_dataset \
--config config.default_config \
--use_gpu \
--train_split 0.5 \
--batch_size 1 \
--modelName Adversarialmodel-pretrain-with-keepdimension-bestDiscriminator/model_45_10000.pt

#Test Accuracy
#baseline(82.40%)
# Epoch.     :  Accuracy
#18          :  0.816063
#20          :  0.816565
#21          :  0.796558
#22          :  0.832556
#23          :  0.821298
#24          :  0.826533
#25.         :  0.804374
#26          :  0.829114
#30          :  0.815705
#35.         :  0.825457
#36          :  0.813697
#37          :  0.811545
#38          :  0.826389
#45.         :  
#50.         :  0.831266
#55          :  0.830907
#59          :  0.831624
