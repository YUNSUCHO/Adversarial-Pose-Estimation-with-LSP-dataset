##takes h the  heatmaps from the hour-glasss network  from the generator POseNET() ,  where the output of the resolution is 64x64
# but  the pose discrimnator has to take a input of 256x256 .
##  SO the last output of the generraotr is upsampled to a size of  256 x256 from 64x64 and then
## concatinated with the pose image  along the channels = 17 in totlal 14 + 3 
## it goes as the input to pose disc
## the PoseNet is  in a eval model, to freeze the weights.





import argparse
import importlib
import random
import os
import shutil
import pickle

import numpy as np
import matplotlib ## was not there # due to tkinter.TclError could not connect to the display
matplotlib.use('svg') ## was not there
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import cv2



from datasets.lsp12000_train import LSP
from datasets.mpii import MPII
from generator import Generator
from discriminator import Discriminator
#from discriminator02 import Discriminator2
from losses import gen_single_loss, disc_single_loss, get_loss_disc, get_loss_disc_bce_with_logits, disc_loss_pose
import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--modelName', required=True, help='name of model; name used to create folder to save model')
parser.add_argument('--config', help='path to file containing config dictionary; path in python module format')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in momentum optimizer')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training, testing')
parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation')
parser.add_argument('--print_every', type=int, default=1000, help='frequency to print train loss, accuracy to terminal')
parser.add_argument('--save_every', type=int, default=200, help='frequency of saving the model')
parser.add_argument('--optimizer_type', default='SGD', help='type of optimizer to use (SGD or Adam or RMS-Prop)')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu for training/testing')
parser.add_argument('--gpu_device', type=int, default=0, help='GPU device which needs to be used for computation')
parser.add_argument('--validation_sample_size', type=int, default=1, help='size of validation sample')
parser.add_argument('--validate_every', type=int, default=50, help='frequency of evaluating on validation set')

parser.add_argument('--path', \
    default='/datasets/lspet_dataset')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=0.9)
parser.add_argument('--heatmap_sigma', type=float, default=3)
parser.add_argument('--occlusion_sigma', type=float, default=1)
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--dataset', type=str, required=True, choices=['mpii', 'lsp'])

args = parser.parse_args()

# initialize seed to constant for reproducibility
np.random.seed(58)
torch.manual_seed(58)
random.seed(58)

# create directory to store models if it doesn't exist
# if it exists, delete the contents in the directory
if (not os.path.exists(args.modelName)):
    os.makedirs(args.modelName)
else:
    shutil.rmtree(args.modelName)
    os.makedirs(args.modelName)

# for handling training over GPU
cpu_device = torch.device('cpu')
fast_device = torch.device('cpu')
if (args.use_gpu):
    fast_device = torch.device('cuda:' + str(args.gpu_device))

# config file storing hyperparameters
config = importlib.import_module(args.config).config



discriminator_model_pose = Discriminator(14 + 3 , config['discriminator']['num_channels'], config['dataset']['num_joints'], config['discriminator']['num_residuals'])

modelpath = torch.load('LSP-keepdimension/LSP12000_VERSION02/model_229_1300.pt') ##--------------> best accurate generator
#model_path_d= torch.load('train-model-16-discriminators/model_10.pt') 
generator_model = modelpath['generator_model']
generator_model= generator_model.eval()
#discriminator_model_pose =model_path_d['discriminator_model_pose'] 

for params in generator_model.parameters():
        params.requires_grad = False
#print(generator_model)
#print(discriminator_model)

# Use dataparallel
generator_model = nn.DataParallel(generator_model)

discriminator_model_pose = nn.DataParallel(discriminator_model_pose)

# Datasets
if args.dataset == 'lsp':
    lsp_train_dataset = LSP(args)
    args.mode = 'val'
    lsp_val_dataset = LSP(args)
# MPII
elif args.dataset == 'mpii':
    lsp_train_dataset = MPII('train')
    lsp_val_dataset = MPII('valid') ## MPII('val') was present originally

# Dataset and the Dataloade
train_loader = torch.utils.data.DataLoader(lsp_train_dataset, batch_size=args.batch_size, shuffle=True)
val_save_loader = torch.utils.data.DataLoader(lsp_val_dataset, batch_size=args.val_batch_size)
val_eval_loader = torch.utils.data.DataLoader(lsp_val_dataset, batch_size=args.val_batch_size, shuffle=True)


pck = metrics.PCK(metrics.Options(256, config['generator']['num_stacks']))

# Loading on GPU, if available
if (args.use_gpu):
    generator_model = generator_model.to(fast_device)
 #   discriminator_model_conf = discriminator_model_conf.to(fast_device)
    discriminator_model_pose = discriminator_model_pose.to(fast_device)

# Cross entropy loss
criterion = nn.CrossEntropyLoss()

# Setting the optimizer
if (args.optimizer_type == 'SGD'):
    optim_gen = optim.SGD(generator_model.parameters(), lr=args.lr, momentum=args.momentum)
    optim_disc = optim.SGD(discriminator_model.parameters(), lr=args.lr, momentum=args.momentum)

elif (args.optimizer_type == 'Adam'):
    optim_gen = optim.Adam(generator_model.parameters(), lr=args.lr) ## added the betas .originally not there 
    optim_disc_pose = optim.Adam(discriminator_model_pose.parameters(), lr=args.lr) ##added the betas.originally not there 
    
#----code added here inplementing rms-prob as an option for optimization--------#
elif (args.optimizer_type == 'RMS'):
    optim_gen =  optim.RMSprop(generator_model.parameters(), lr=args.lr)
    optim_disc=  optim.RMSprop(discriminator_model.parameters(), lr=args.lr)
else:
    raise NotImplementedError
    
    

    





gen_losses = []

disc_losses = [] 

val_gen_losses = []

val_disc_losses = []

pose_fake =[]

pose_real =[]

mean_accuracy =[]
mean_count    =[]

def evaluate_model(args, epoch, val_loader, fast_device, generator_model, discriminator_model):
    os.makedirs(os.path.join(args.modelName, str(epoch)))
    gen_loss = 0.0
    disc_loss = 0.0
    disc_loss_pose = 0.0
    all_images = None
    all_outputs = []
    all_ground_truth = {}
    for i, data in enumerate(val_loader):
        print("index :",i)
        if i>=100 : break  #---------------------------------------> validation samples 
        images = data['image']
        if (all_images is None):
            all_images = images.numpy()
        else:
            all_images = np.concatenate((all_images, images.numpy()), axis=0)
        ground_truth = {}
        ground_truth['heatmaps'] = data['heatmaps']
        ground_truth['occlusions'] = data['occlusions']
        if (len(all_ground_truth) == 0):
            for k, v in ground_truth.items():
                all_ground_truth[k] = ground_truth[k].numpy()
        else:
            for k, v in ground_truth.items():
                all_ground_truth[k] = np.concatenate((all_ground_truth[k], ground_truth[k].numpy()), axis=0)
        
        if (args.use_gpu):
            images = images.to(fast_device)
            ground_truth['heatmaps'] = ground_truth['heatmaps'].to(fast_device)
            ground_truth['occlusions'] = ground_truth['occlusions'].to(fast_device)

        with torch.no_grad():
            outputs = generator_model(images)
            cur_gen_loss_dic = gen_single_loss(ground_truth, outputs, discriminator_model, mode=args.loss)
            cur_disc_loss_dic = disc_single_loss(ground_truth, outputs, discriminator_model)

            cur_gen_loss = cur_gen_loss_dic['loss']
            cur_disc_loss = cur_disc_loss_dic['loss']

            gen_loss += cur_gen_loss
            disc_loss += cur_disc_loss

            if (len(all_outputs) == 0):
                for output in outputs:
                    all_outputs.append(output.to(cpu_device).numpy())
            else:
                for i in range(len(outputs)):
                    all_outputs[i] = np.concatenate((all_outputs[i], outputs[i].to(cpu_device).numpy()), axis=0)

    with open(os.path.join(args.modelName, str(epoch), 'validation_outputs.dat'), 'wb') as f:
        pickle.dump((all_images, all_ground_truth, all_outputs), f)
        
    return gen_loss, disc_loss

val_pos = 0
for epoch in range(args.epochs):
    print('epoch:', epoch)

    if (epoch > 0 and epoch % args.save_every == 0):
        torch.save({'discriminator_model_pose': discriminator_model_pose,
                    'criterion': criterion,                                         
                    'optim_disc_pose': optim_disc_pose}, os.path.join(args.modelName, 'model_' + str(epoch) + '.pt'))
        #val_gen_loss, val_disc_loss = evaluate_model(args, epoch, val_save_loader, fast_device, generator_model, discriminator_model_pose)
        #val_gen_losses.append(val_gen_loss)
        #val_disc_losses.append(val_disc_loss)

    epoch_gen_loss = 0.0
    epoch_disc_loss = 0.0
    print("length of training datasets: ",len(train_loader))
    print("length of validation datasets: ",len(val_eval_loader))
    for i, data in enumerate(train_loader):
        #optim_gen.zero_grad()
        #optim_disc_conf.zero_grad()
        #optim_disc_pose.zero_grad()
        
        images = data['image']
        ground_truth = {}
        ground_truth['heatmaps'] = data['heatmaps']
        ground_truth['occlusions'] = data['occlusions']
        if (args.use_gpu):
            images = images.to(fast_device)
            ground_truth['heatmaps'] = ground_truth['heatmaps'].to(fast_device)
            ground_truth['occlusions'] = ground_truth['occlusions'].to(fast_device)

    ########################  Check and complete the code here ######## Forward pass and calculate losses here ################
        
        
        
       
            
       
            
            
       
        print("---------------------pose Discriminator training-------------------------------------------------")
        print('epoch no :', epoch)
        optim_disc_pose.zero_grad()
        optim_gen.zero_grad()

        outputs = generator_model(images)
        #print(type(outputs))
        
        outputs_d = [nn.Upsample(scale_factor=256 / 64, mode='nearest')(outputs[-1])]
        
        #print(type(outputs))
        #out_p  = outputs[-1]
        #print(out_p)
        #print(type(outputs_d))
        outputs[-1]  = outputs[-1][:, : config['dataset']['num_joints']]
        _ , _ , p_fake = pck.StackedHourGlass(outputs, ground_truth['occlusions'])
        
        
        cur_disc_loss_dic = disc_loss_pose(ground_truth, outputs_d, images , discriminator_model_pose, detach=False, p_fake = p_fake )

        
        cur_disc_loss = cur_disc_loss_dic['loss']
        
        pose_real_loss     = cur_disc_loss_dic['pose_disc_real']
        pose_fake_loss     = cur_disc_loss_dic["pose_disc_fake"]



        loss_pose = cur_disc_loss
        
        #print("pose conf loss {}".format(loss_pose))

        loss_pose.backward()


        optim_disc_pose.step()


        #------------------------------------------------------------------------------------------------------------#    
            
            

        
        cur_disc_loss = loss_pose.item()
        
        disc_losses.append(float(cur_disc_loss))
       # epoch_gen_loss += cur_gen_loss
       # epoch_disc_loss += cur_disc_loss
        
        with open(os.path.join(args.modelName, 'stats.bin'), 'wb') as f:
            pickle.dump((disc_losses, pose_real , pose_fake), f)

        if i % args.print_every == 0:
            print("Train iter: %d, discriminator loss : %f" % (i, disc_losses[-1]))

            plt.clf()
            plt.imshow(np.sum(outputs[-1].detach().cpu().numpy()[0][0 : 14], axis=0))
            plt.savefig('trainingImages/train_output_exp23.png')
            
            
            plt.clf()
            plt.imshow(np.sum(ground_truth['heatmaps'].detach().cpu().numpy()[0], axis=0))
            plt.savefig('trainingImages/train_gt_exp23.png')

            plt.clf()
            print("image shape_exp16: ", images.shape)
            plt.imshow((images.detach().cpu().numpy()[0].transpose(1, 2, 0) * 128 + 128).astype(np.uint8)) ## originally it was there , replaced by below line
            #plt.imshow((images.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255)
            plt.savefig('trainingImages/train_img_exp23.png')
        # Save model
        ######################################################################################################################
        if i > 0 and i % 8000 == 0:
            # Saving the model and the losses
            torch.save({'discriminator_model_pose': discriminator_model_pose,
                        'optim_disc_pose': optim_disc_pose }, \
                        os.path.join(args.modelName, 'model_{}_{}.pt'.format(epoch, i)))

        
        mean_eval_avg_acc, mean_eval_cnt = 0.0, 0.0
        if i % args.validate_every == 0:
            for j, data in enumerate(val_eval_loader):
                if (j == args.validation_sample_size):
                    break
                images = data['image']
        
                ground_truth = {}
                ground_truth['heatmaps'] = data['heatmaps']
                ground_truth['occlusions'] = data['occlusions']
        
                if (args.use_gpu):
                    images = images.to(fast_device)
                    ground_truth['heatmaps'] = ground_truth['heatmaps'].to(fast_device)
                    ground_truth['occlusions'] = ground_truth['occlusions'].to(fast_device)

                with torch.no_grad():
                    outputs = generator_model(images)
                    outputs[-1] = outputs[-1][:, : config['dataset']['num_joints']]
                    eval_avg_acc, eval_cnt, _ = pck.StackedHourGlass(outputs, ground_truth['heatmaps'])
                    mean_eval_avg_acc += eval_avg_acc
                    mean_eval_cnt += eval_cnt
#------------------append the calulated accuracy  to a list to visualize the the pck values over each ititation-------------#            
                
            #mean_accuracy.append(mean_eval_avg_acc)
            #mean_count.append(mean_eval_cnt)
            
            print("Validation avg acc: %f, eval cnt: %f" % (mean_eval_avg_acc, mean_eval_cnt))
            #with open(os.path.join(args.modelName, 'pckstats.bin'), 'wb') as p:
            #     pickle.dump((mean_accuracy, mean_count), p)
#--------------------------------------------------------------------------------------------------------------------------#

    #epoch_gen_loss /= len(lsp_train_dataset)
    #epoch_disc_loss /= len(lsp_train_dataset)

    #print('Epoch train gen loss: %f' % (epoch_gen_loss))
    #print('Epoch train disc loss: %f' % (epoch_disc_loss))
    
    #with open(os.path.join(args.modelName, 'stats.bin'), 'wb') as f:
    #        pickle.dump((disc_losses, gen_losses, val_disc_losses, val_gen_losses), f)

#with open(os.path.join(args.modelName, 'stats.bin'), 'wb') as f:
#    pickle.dump((disc_losses, gen_losses, val_disc_losses, val_gen_losses), f)


# Plotting the loss function
plt.plot(loss.detach().cpu().numpy()) # detach().cpu() was not there
plt.savefig(os.path.join(args.modelName, 'loss_graph.pdf')) ##orginally it was there .the line below is being added , due to error 
#plt.savefig(os.path.join(args.modelName, 'loss_graph.png'))
plt.clf()