# Adversarial-Pose-Enstimation



Investigation on adversarial learning for pose prediction in Leeds Sports Pose Dataset


Pytorch implementation of chen et al. "Adversarial PoseNet" for landmark localization on digital images.
The architecture was  proposed by [Yu Chen, Chunhua Shen, Xiu-Shen Wei, Lingqiao Liu, Jian Yang](https://scholar.google.com/citations?user=IWZubqUAAAAJ&hl=zh-CN) in 
[Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation](https://arxiv.org/pdf/1705.00389). 



## Lanmark localization 


<img src="https://github.com/YUNSUCHO/Adversarial-Pose-Enstimation/blob/main/README/Lanmark%20localization%20.png" width="300px"/>

Before the augmentation techniques applied on the dataset

<img src="https://github.com/YUNSUCHO/Adversarial-Pose-Enstimation/blob/main/README/Lanmark%20localization%20(after%20augmentation).png" width="300px"/>

After the augmentation techniques applied on the dataset


##  Results Visualization
The results of this implementation:

### Adversarial PoseNet(In Adversarial setup using GAN framework):
<img src="https://github.com/YUNSUCHO/Adversarial-Pose-Enstimation/blob/main/README/original.png" width="200px"/><img src="https://github.com/YUNSUCHO/Adversarial-Pose-Enstimation/blob/main/README/Ad%20mode.png" width="200px"/>



### Stack-hour-glass Network(In supervised setup):
<img src="https://github.com/YUNSUCHO/Adversarial-Pose-Enstimation/blob/main/README/original.png" width="200px"/><img src="https://github.com/YUNSUCHO/Adversarial-Pose-Enstimation/blob/main/README/supervised%20mod.png" width="200px"/>

### localization rate of diffent setups on the test split:

The PCK@0.2 metrics has been used to measure the performance of the proposed model.

<img src="https://github.com/YUNSUCHO/Adversarial-Pose-Enstimation/blob/main/README/Screen%20Shot%202020-10-29%20at%205.27.18%20PM.png" width="300px"/>

<img src="https://github.com/YUNSUCHO/Adversarial-Pose-Enstimation/blob/main/README/Screen%20Shot%202020-10-29%20at%205.26.49%20PM.png" width="300px"/>


## Main Prerequisites
- pytorch
- OpenCV
- Numpy
- Scipy-images
- ```The list of dependencies can be found in the the requirements.txt file. Simply use pip install -r requirements-pytorch-posenet.txt to install them.```


## Getting Started
### Installation
- Install Pytorch from https://pytorch.org/get-started/locally/
- Clone this repository:
```bash
git clone https://github.com/YUNSUCHO/Adversarial-Pose-Enstimation.git
```


## Training and Test Details
To train a model, run any of the .sh file starting with "train". For example :  
```bash
Adversarialmodel-pretrain-with-keepdimension.sh 
```
- A bash file has following configurations, that one can change 
```
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
```
Models are saved to `./trainmodel/` (can be changed using the argument --modelName in .sh file).  

To test the model,
```bash
test-Adversarialmodel-pretrain-with-keepdimension.sh
```

## Datasets


- ` Leeds Sports Pose Dataset`: The LSP-extended dataset contains 10,000 annotated images in the RGB nature of most sportspeople. Every image that have different sizes, since it is not quadratic. The images have scaled such that the most prominent person in the image is roughly 150 pixels in length. Each image has been annotated with 14 co-ordinate joints locations.The available body joints in the LSP-extended dataset are right ankle, right knee, right hip, left hip, left knee, left ankle, right wrist, right elbow, right shoulder, left shoulder, left elbow, left wrist, neck, head top.,



<img src="https://github.com/YUNSUCHO/Adversarial-Pose-Enstimation/blob/main/README/original%20RGB%20.png" width="400px"/> 


## Reference
- The pytorch implementation of stacked-hour-glass, https://github.com/princeton-vl/pytorch_stacked_hourglass
- The pytorch implementation of self-adversarial pose estimation, https://github.com/roytseng-tw/adversarial-pose-pytorch
- The torch implementation of self-adversarial pose estimationh , https://github.com/dongzhuoyao/jessiechouuu-adversarial-pose


