CUDA_VISIBLE_DEVICES=1
python testImages.py \
--mode train \
--path lsp_dataset/lsp_dataset \
--config config.default_config \
--use_gpu \
--train_split 0.25 \
--batch_size 1 \
--modelName Adversarialmodel-pretrain-with-keepdimension/model_57_10000.pt



#Check training accuracy of the model ( Adversarial-Pose-Estimation.lsp.V.4/Adversarialmodel-pretrain-with-keepdimension) with the subset of training image(500images from LSP)

#Epoch : Training Accuracy
#0     : 0.868376
#1     : 0.875967
#2     : 0.863076
#3     : 0.869665
#4     : 0.862647
#5     : 0.860069
#6     : 0.837009
#7     : 0.881839 
#8     : 0.894300
#9     : 0.900745
#10    : 0.836580
#11    : 0.891435
#12    : 0.874535
#13    : 0.860642
#14    : 0.886995
#15    : 0.880407
#16    : 0.886995
#17    : 0.904469
#18    : 0.907190
#19    : 0.885420
#20    : 0.911343
#21    : 0.905042
#22    : 0.914065
#23    : 0.912203
#24    : 0.908622
#25    : 0.912776
#26    : 0.907763
#27    : 0.910627
#28    : 0.912776
#29    : 0.885563
#30    : 0.913635
#31    : 0.918648
#32    : 0.907763
#33    : 0.907190
#34    : 0.911057
#35    : 0.926239
#36    : 0.924091
#37    : 0.910341
#38    : 0.928101
#39    : 0.924234
#40    : 0.931109
#41    : 0.929103
#42    : 0.898883
#43    : 0.932684
#44    : 0.919221
#45    : 0.927385
#46    : 0.927098
#47    : 0.927528
#48    : 0.929820
#49    : 0.916500
#50    : 0.932684
#51    : 0.938127
#52    : 0.927098
#53    : 0.925666
#54    : 0.933257
#55    : 0.941134
#56    : 0.930679
#57    : 0.935692
#last(58)  :0.934976

