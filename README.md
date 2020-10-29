# Adversarial-Pose-Enstimation



Experiments on adversarial learning for pose prediction in digital images(Leeds Sports Pose Dataset)


Pytorch implementation of chen et al. "Adversarial PoseNet" for landmark localization on digital images.
The architecture was  proposed by [Yu Chen, Chunhua Shen, Xiu-Shen Wei, Lingqiao Liu, Jian Yang](https://scholar.google.com/citations?user=IWZubqUAAAAJ&hl=zh-CN) in 
[Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation](https://arxiv.org/pdf/1705.00389). 



## Lanmark localization 
<img src="README/Screen Shot 2020-03-31 at 9.34.51 PM.png" width="500px"/>



##  Results Visualization
The results of this implementation:

### Adversarial PoseNet:
<img src="testresults-1/Adversarial-1/results_116.png" width="500px"/>




### Stack-hour-glass Network(supervised setup):
<img src="testresults-1/baseline-1/results_120.png" width="500px"/> 

### localization rate of diffent setups on the test split:

<img src="readmeimages/result_hist.png" width="400px"/><img src="readmeimages/result_table.png" width="400px"/>





## Main Prerequisites
- pytorch
- OpenCV
- Numpy
- Scipy-images
- ```The list of dependencies can be found in the the requirements.txt file. Simply use pip install -r requirements.txt to install them.```


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
python trainmodeladversarial-pos-conf-exp24.py \
--path handmedical \
--modelName trainmodel \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--lr .0002 \
--print_every 100 \
--train_split 0.804 \
--loss mse \
--optimizer_type Adam \
--epochs 50 \
--dataset  'medical' 
@@@ need to change@@
```
Models are saved to `./trainmodel/` (can be changed in the --modelName).  

To test the model,
```bash
test-Adversarialmodel-pretrain-with-keepdimension.sh
```

## Datasets


- ` Leeds Sports Pose Dataset`: The LSP-extended dataset contains 10,000 annotated images in the RGB nature of most sportspeople. Every image that have different sizes, since it is not quadratic. The images have scaled such that the most prominent person in the image is roughly 150 pixels in length. Each image has been annotated with 14 co-ordinate joints locations.The available body joints in the LSP-extended dataset are right ankle, right knee, right hip, left hip, left knee, left ankle, right wrist, right elbow, right shoulder, left shoulder, left elbow, left wrist, neck, head top.,

["Detection and Localization of Landmarks in the Lower Extremities Using an Automatically Learned 
Conditional Random Field](https://www.researchgate.net/publication/319634278_Detection_and_Localization_of_Landmarks_in_the_Lower_Extremities_Using_an_Automatically_Learned_Conditional_Random_Field)


<img src="readmeimages/lowerleg_greyscale.png" width="220px"/><img src="readmeimages/lowerleg.png" width="220px"/>   <img src="readmeimages/lowerleg_annotated.png" width="300px"/>


## Reference
- The pytorch implementation of stacked-hour-glass, https://github.com/princeton-vl/pytorch_stacked_hourglass
- The pytorch implementation of self-adversarial pose estimation, https://github.com/roytseng-tw/adversarial-pose-pytorch
- The torch implementation of self-adversarial pose estimationh , https://github.com/dongzhuoyao/jessiechouuu-adversarial-pose


