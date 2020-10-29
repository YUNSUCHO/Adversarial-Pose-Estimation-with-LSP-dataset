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
import datasets.img as I
# from PIL import Image
from random import randint
import time



parser = argparse.ArgumentParser()

parser.add_argument('--path', \
    default='/home/ycho/Adversarial-Pose-EstimationV.2/lsp_dataset/')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=.9167)   #0.9142
parser.add_argument('--heatmap_sigma', type=float, default=3)
parser.add_argument('--occlusion_sigma', type=float, default=1)

class LSP(Dataset):
    '''
    LSP dataset
    '''
    def __init__(self, cfg):
        # Path = dataset path, mode = train/val
        self.path = cfg.path
        self.path_s= '/home/ycho/Adversarial-Pose-Estimation.lsp.V.4/lspet_dataset'
        self.mode = cfg.mode
        self.crop_size = cfg.crop_size
        self.train_split = cfg.train_split
        self.heatmap_sigma = cfg.heatmap_sigma
        self.occlusion_sigma = cfg.occlusion_sigma
        
        self.rotate = 30        
        self.inputRes = 256
        
        self.out_size   = 256
        self.out_size_b = 64
        assert self.mode in ['train', 'val'], 'invalid mode {}'.format(self.mode)
        assert cfg.train_split > 0 and cfg.train_split < 1, 'train_split should be a fraction'
        self._get_files()

    def _get_files(self):
        # Get files for train/val
        self.files = sorted(glob(join(self.path_s, 'images/*.jpg'))) + sorted(glob(join(self.path, 'images/*.jpg')))
        print(len(self.files))
        #print(self.files)
        
        self.annot = np.concatenate((loadmat(join(self.path_s, 'joints.mat'))['joints'],loadmat(join(self.path, 'joints.mat'))
                                     ['joints'].transpose(1 , 0 , 2)),  axis =2)
        
        print("length of annotation:",(self.annot).shape)

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
        file_name = self.files[idx]
        #print("index",file_name)
        start = time.process_time()
        img = cv2.imread(file_name)
       
                    
        
        c = np.array([img.shape[0]/2 , img.shape[1]/2.5])
        #----------------------------------------------
        
        if img.shape[0] >= img.shape[1]:
            s = img.shape[0]*1.5
        elif img.shape[0] <= img.shape[1]:
        #----------------------------------------------
            s = img.shape[1]*1.5
        #print("s  :" ,s )
        #===========================================================
        #s = 400
        r = 0
        s = s   * (1.1 ** I.Rnd(1.75))#1.5 #(randint(1, 2))
        r = 0 if np.random.random() < 0.6 else I.Rnd(30)
        
        
        
        
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print("shape of the image:" , img.shape)
        #crop_image = cv2.resize(image, (self.crop_size, self.crop_size))
        crop_image = I.Crop(img, c, s, r, self.inputRes) / 255.
        print(time.process_time() - start)
        
        
        
        
        # Read annotations
        annot = self.annot[:,:, idx]+ 0.0#annot = self.annot[:, :, idx]
        
        # Read annotations
        annot_b = self.annot[:, :, idx] + 0.0
        
        #print(annot_b)
        
        
        #================================= Generate 64 heatmaps============================================#
        x = range(self.out_size)  # x = range(self.crop_size) --- new
        xx, yy = np.meshgrid(x, x)
        
        
        # ==================================Generate  256 heatmaps==========================================#
        m = range(self.out_size_b)  # x = range(self.crop_size) --- new
        mm, nn = np.meshgrid(m, m)
        
        #new heatmaps = np.zeros((annot.shape[0], self.crop_size, self.crop_size))
        
        heatmaps = np.zeros((14, self.out_size, self.out_size))
        
        #new occlusions = np.zeros((annot.shape[0], self.crop_size, self.crop_size))
        
        occlusions = np.zeros((14, self.out_size_b, self.out_size_b)) 
        
        #print(annot_b.shape[0])
        #======================================================================================================#
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
               
                
                
                occlusions[joint_id] = I.DrawGaussian(occlusions[joint_id], np.array([ m_c , n_c]), 1, 0.5 if self.out_size_b==32 else -1)
                
                
                

        #-------------------------------------------------------------  
        if np.random.random() < 0.5:
            crop_image = I.Flip(crop_image)
            occlusions = I.ShuffleLR_LSP(I.Flip(occlusions))
            heatmaps = I.ShuffleLR_LSP(I.Flip(heatmaps))

        img = torch.Tensor(crop_image)
        #-------------------------------------------------------------
        print("end :" , time.process_time() - start)
        return {
            # image is in CHW format
            'image':img,  # torch.Tensor(crop_image.transpose(2, 0, 1)),#,/255.,
            #'kp_2d': torch.Tensor(annot),
            'heatmaps': torch.Tensor(heatmaps*1.2),
            'occlusions': torch.Tensor(occlusions*1.2),
            # TODO: Return heatmaps
        }

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = LSP(args)
    print("the length of the dataset" , len(dataset))
    for i in range(len(dataset)): 
            data = dataset.__getitem__(i)
            plt.clf()
            plt.figure(figsize=(20,20))
            print(data['image'].min())
            print(data['image'].max())
            plt.subplot(1, 3, 1)
            #plt.imshow(data['image'].numpy().transpose(1, 2, 0)/255.0)  ## originally it was  there ,noramlized images are produced
            plt.imshow((data['image'].numpy().transpose(1, 2, 0)*255).astype(np.uint8))
            #plt.scatter(data['kp_2d'][:, 0].numpy(), data['kp_2d'][:, 1].numpy(), c=data['kp_2d'][:, 1])

            plt.subplot(1, 3, 2)
            plt.imshow(data['heatmaps'].numpy().sum(0))
            print(data['heatmaps'].numpy().sum(0).min())
            print(data['heatmaps'].numpy().sum(0).max())

            plt.subplot(1, 3, 3)
            plt.imshow(data['occlusions'].numpy().sum(0))

            plt.show()

