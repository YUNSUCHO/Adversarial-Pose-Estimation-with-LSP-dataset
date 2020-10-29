'''
LSP Dataset
'''
from os.path import join
import argparse

from glob import glob
import cv2
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import numpy as np
# from PIL import Image
from torchvision.transforms import *
import datasets.img as I

parser = argparse.ArgumentParser()

parser.add_argument('--path', \
    default='/home/ycho/Adversarial-Pose-EstimationV.2/lsp_dataset')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=.25)# amount of images that i want to use for train
parser.add_argument('--heatmap_sigma', type=float, default=2)
parser.add_argument('--occlusion_sigma', type=float, default=1)

class LSP(Dataset):
    '''
    LSP dataset
    '''
    def __init__(self, cfg):
        # Path = dataset path, mode = train/val
        self.path = cfg.path
        self.mode = cfg.mode
        self.crop_size = cfg.crop_size
        self.train_split = cfg.train_split
        self.heatmap_sigma = cfg.heatmap_sigma
        self.occlusion_sigma = cfg.occlusion_sigma
        self.out_size   = 256
        self.out_size_b = 64
        self.inputRes = 256
        assert self.mode in ['train', 'val'], 'invalid mode {}'.format(self.mode)
        assert cfg.train_split > 0 and cfg.train_split < 1, 'train_split should be a fraction'
        self._get_files()

    def _get_files(self):
        # Get files for train/val
        self.files = sorted(glob(join(self.path, 'images/*.jpg')))
        #print(self.files)
        self.annot = loadmat(join(self.path, 'joints.mat'))['joints'].transpose(1 ,0 ,2) 
        #print(self.annot)

    def __len__(self):
        # Return length
        if self.mode == 'train':
            return int(self.train_split * len(self.files))
        else:
            return len(self.files) - int(self.train_split * len(self.files))


    def __getitem__(self, idx):
        # if validation, offset index
        #idx =3
        if self.mode == 'val':
            idx += int(self.train_split * len(self.files))

        # Get the i'th entry
        print("index :", idx)
        file_name = self.files[idx]
        print(file_name)
        img = cv2.imread(file_name)
        #print(img.shape)
        #===========================================================
        c = np.array([img.shape[0]/2.5 , img.shape[1]/1.2]) #(c  is the center cordinate of the body) #image.shape[0]:height, image.shape[1]:width
        if img.shape[0] >= img.shape[1]:
            s =img.shape[0]*1.3                          #(s is the height or scale of the body in the image) 
        if img.shape[0] <= img.shape[1]:
            c = np.array([img.shape[0]/2 , img.shape[1]/2.5])
            s =img.shape[0]*1.5
        #===========================================================
        
        
        
        
        #if img.shape[0] < img.shape[1]:
        #    s = s = img.shape[1]*1.2
        #s = 400
       
        s = s  #*2 bigger the number the smaller the image # * (1.1 ** I.Rnd(1.2))#1.5 #(randint(1, 2))
        r = 0 #if np.random.random() < 0.6 else I.Rnd(20)         #(r is the rotation applied )
        
        
        # image = image.resize((self.crop_size, self.crop_size))
        #image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
        #image = (image)/255;                                    #(image - 128.0)/ 128;
        
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #crop_image = cv2.resize(image, (self.crop_size, self.crop_size))
        crop_image = I.Crop(img, c, s, r, self.inputRes) / 255. 
        
        
        
        
        # Read annotations
        annot = self.annot[:,:, idx]+ 0.0#annot = self.annot[:, :, idx]
        #print(annot)
        
        # annot = K * 3
        ####annot[:, :2] = annot[:, :2] * np.array(\
        ####    [[self.out_size*1.0/img.shape[0], self.out_size*1.0/img.shape[1]]])
        
        #print(annot)
        
        
        #crop_image_b = cv2.resize(image, (self.crop_size_b, self.crop_size_b))
        

        # Read annotations
        annot_b = self.annot[:, :, idx] + 0.0
        x = range(self.out_size)  # x = range(self.crop_size) --- new
        xx, yy = np.meshgrid(x, x)
        
        
        # Generate  256 heatmaps
        m = range(self.out_size_b)  # x = range(self.crop_size) --- new
        mm, nn = np.meshgrid(m, m)
        
        #new heatmaps = np.zeros((annot.shape[0], self.crop_size, self.crop_size))
        
        heatmaps = np.zeros((14, self.out_size, self.out_size))
        
        #new occlusions = np.zeros((annot.shape[0], self.crop_size, self.crop_size))
        
        occlusions = np.zeros((14, self.out_size_b, self.out_size_b)) 
        #print(annot_b.shape[0])
        # Annotate heatmap
        for joint_id in range(annot.shape[0]):
            #print("joint id ", annot[joint_id][0])
            if annot[joint_id][0] >0: 
                x_c1, y_c1, vis = annot[joint_id] + 0

                annot_pt= I.Transform(np.array([x_c1 , y_c1]), c, s, r, 256)
                x_c, y_c = annot_pt

                m_c, n_c, vis = annot[joint_id] + 0

                annot_b_pt= I.Transform(np.array([x_c1 , y_c1]), c, s, r, 64)

                m_c , n_c  =  annot_b_pt
                

                #print("vis :",x_c1, y_c)
                
                heatmaps[joint_id] =I.DrawGaussian(heatmaps[joint_id], np.array([x_c, y_c]), 1, 0.5 if self.out_size==32 else -1)
                #np.exp(-1*((x_c - xx)**2 + (y_c - yy)**2)/(self.heatmap_sigma**2))
                
                
                #I.DrawGaussian(heatmaps[joint_id], np.array([x_c, y_c]), 2, 0.5 if self.out_size==32 else -1)  

                #np.exp(-1*((x_c - xx)**2 + (y_c - yy)**2)/(self.heatmap_sigma**2))
                #occlusions[joint_id] =np.exp(-0.5*((x_c - xx)**2 + (y_c - yy)**2)/(self.occlusion_sigma**2))
                
                
                occlusions[joint_id] = I.DrawGaussian(occlusions[joint_id], np.array([ m_c , n_c]), 1, 0.5 if self.out_size_b==32 else -1)
                
        #---------------
        
        #if occlusions[n, 3, 0] > 1 and target[n, 8, 0] > 1:  
                #    rt_sh  =target[n, 8, :2] 
                #    lf_hip =target[n, 3, :2]
                #    normalize = np.linalg.norm(lf_hip - rt_sh)
                #    normalize = np.array((normalize , normalize)) 
                #    print(normalize)
                
        
        #print("eludian distance: left shoulder and right hip" , dis)
        
        #lefthip - rightshoulder
        #---------------
        return {
            # image is in CHW format
            'image': torch.Tensor(crop_image),#,/255.,
            #'kp_2d': torch.Tensor(annot),
            'heatmaps': torch.Tensor(occlusions*1.2),
            'occlusions': torch.Tensor(occlusions*1.2),
            #'distance' : torch.Tensor([dis])
            # TODO: Return heatmaps
        }

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = LSP(args)
    print(len(dataset))
    d = 0
    c = 0
    array = []
    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        plt.clf()
        #if data['distance'].numpy()[0] == 0.0:
        c+=1
        print("image no :" , c)
        #print("No right left", c)
        #print("eludian distance: left shoulder and left hip",data['distance'].numpy()[0])
        #d += data['distance'][0]
        #if data['distance'][0] != 0:
            
         #   array.append(data['distance'][0])
        plt.figure(figsize = (20 , 10))
        #print("median",np.median(array))
        #print("d :" , float(d))
        #print(data['image'].min())
        #print(data['image'].max())
        #plt.figure(figsize=(20,20))
        plt.subplot(1, 3, 1)
        plt.imshow(data['image'].numpy().transpose(1, 2, 0)/255.0)  ## originally it was  there ,noramlized images are produced
        plt.imshow((data['image'].numpy().transpose(1, 2, 0)*255).astype(np.uint8))
        #plt.scatter(data['kp_2d'][3, 0].numpy(), data['kp_2d'][3, 1].numpy(), c="yellow")#data['kp_2d'][:, 1]

        plt.subplot(1, 3, 2)
        plt.imshow(data['heatmaps'].numpy().sum(0))
        #print(data['heatmaps'].numpy().sum(0).min())
        #print(data['heatmaps'].numpy().sum(0).max())

        plt.subplot(1, 3, 3)
        plt.imshow(data['occlusions'].numpy().sum(0))

        plt.show()

   