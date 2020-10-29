'''
file containing loss functions
'''
import torch
from torch.nn import functional as F
import torch.nn  as nn
import numpy as np 

def get_loss_recon(out, inp, mode):  # --------> out = predicted inp ------> ground_truth 
	'''
	Get the reconstruction loss
	'''
	if mode == 'mse':
		#loss = ((out - inp)**2).mean
		#print("loss",loss)
		loss = nn.MSELoss()
		loss = loss(out , inp) 
		#loss = loss.mean()
        
	elif mode == 'mae':
		loss = nn.L1Loss()
		loss = loss(out , inp)
        
	elif mode == 'bce_skew':
		# here target = out
		# inp = ground truth
		mask = (inp > 0).float()
		f = mask.mean()
		ratio = (1-f)/f
		loss = F.binary_cross_entropy(out, inp, weight=(ratio**mask)).mean()
	elif mode == 'bce':
		#loss = F.binary_cross_entropy(out, inp)
		loss =nn.BCELoss()
		loss =loss(out , inp)        
	elif mode == 'bce_l' :
		loss = F.binary_cross_entropy_with_logits(out, inp)        
	else:
		raise NotImplementedError
	return loss


def get_loss_disc(output, discriminator, detach=False, real=True, eps=1e-5):
	'''
	Get discriminator loss
	'''
	outs = output.detach() if detach else output                
	if real:
		loss = -torch.log(eps + discriminator(outs)).mean()
	else:
		loss = -torch.log(eps + 1 - discriminator(outs)).mean() 
	return loss



def get_loss_disc_bce_with_logits(output, discriminator, detach=False, real=True, ground_truth=None, p_fake = None):
	'''
	Get discriminator loss
    bce, bce_w_logits, 
	'''
	p_fake = p_fake    
	outs =output.detach() if detach else output     
	discriminator_out = discriminator(outs)
	batch_size = discriminator_out.size()
	gt_labels = ground_truth
    
	#print("batch_size :" , batch_size) -------------------> (batchsize, 14 , 256, 256)
	#print("disc output",(discriminator_out).size())    
    
	device = torch.device('cuda:0')
   
    
      
	if real:
		#print("real label is being calculated") 
		labels = torch.ones(batch_size) # real labels = 1
		#ground_labels =gt_labels  ---------------------------->   theses are ground truth 
		#print("ground_label :",ground_labels.size())        
		labels = labels.to(device)
    
		criterion0 = nn.BCEWithLogitsLoss()
		criterion1 = nn.BCELoss()
#		print("output of the discriminator for real images  :",discriminator_out)           
		loss = F.binary_cross_entropy(discriminator_out,labels)
		#loss = criterion1(discriminator_out, labels)
	else:    #-------------------------------------------------------------> else-> not real 
		#print("fake label is being calculated")        
		labels = torch.zeros(batch_size) # fake labels = 0
		epsilon= torch.Tensor([p_fake])
		epsilon= epsilon.to(device)
#		print("p_fake value :" , epsilon)      
        
		labels = labels.to(device)
#		print("output of the discriminator for fake images  :",discriminator_out)   
		#criterion0 = nn.BCEWithLogitsLoss()
		#criterion1 =   nn.BCELoss()
    
		loss = F.binary_cross_entropy(torch.abs(discriminator_out - epsilon), labels )             
	return loss



#----------------------------------------lsgan discriminator objective---------------------------------------------#
def get_loss_disc_mse_lsgan(output, discriminator, detach=False, real=True, eps=1e-5):
	'''
	Get discriminator loss
    mse loss, 
	'''
	outs = output.detach() if detach else output 
	discriminator_out = discriminator(outs)
	#print("disc output",(discriminator_out).size())
   
                                                                            #not used in the script
      
	if real:# for real mean squared error.
		print("real label is being calculated") 
 # how close is the produced output from being "real"?
		loss = torch.mean((discriminator_out-1)**2)            

	else:# for fake mean squared error.
		print("fake label is being calculated")            
		loss = torch.mean(discriminator_out**2)          
	return loss
#-------------------------------------------------------------------------------------------------------------------#

def gen_loss(ground_truth,
			 outputs,
			 images,
			 pose_discriminator,
			 conf_discriminator,
			 mode='mse',
			 alpha=1/220.,
			 beta=1/180.
			):
	'''
	Get generator loss
	'''
	gt_maps = torch.cat([ground_truth['heatmaps'], ground_truth['occlusions']], 1)
	gt_w_images = torch.cat([gt_maps, images], 1)
	loss_recon = 0.0
	loss_conf_disc = 0.0
	loss_pose_disc = 0.0
	for output in outputs:
		# MSE loss, confidence loss, pose loss
		loss_recon = loss_recon + get_loss_recon(output, gt_maps, mode)
		loss_conf_disc = loss_conf_disc + get_loss_disc(output, conf_discriminator)
		loss_pose_disc = loss_pose_disc + get_loss_disc(torch.cat([output, images], 1), pose_discriminator)

	loss_recon = loss_recon / len(outputs)
	loss_conf_disc = loss_conf_disc / len(outputs)
	loss_pose_disc = loss_pose_disc / len(outputs)
	# Add discriminator loss
	return {
		'loss': loss_recon + alpha*loss_conf_disc + beta*loss_pose_disc,
		'recon': loss_recon,
		'conf_disc': loss_conf_disc,
		'pose_disc': loss_pose_disc,
	}


def disc_loss(ground_truth,
			 outputs,
			 images,
			 pose_discriminator,
			 conf_discriminator,
			 alpha=1/220.0,
			 beta=1/180.0
			):
	'''
	Get discriminator loss
	'''
	gt_maps = torch.cat([ground_truth['heatmaps'], ground_truth['occlusions']], 1)
	gt_w_images = torch.cat([gt_maps, images], 1)
	loss_conf_real = get_loss_disc(gt_maps, conf_discriminator)
	loss_pose_real = get_loss_disc(gt_w_images, pose_discriminator)
	# False for generator
	for output in outputs:
		loss_conf_disc = loss_conf_disc + \
			get_loss_disc(output, conf_discriminator, detach=False, real=False)
		loss_pose_disc = loss_pose_disc + \
			get_loss_disc(torch.cat([output, images], 1), pose_discriminator, detach=False, real=False)

	loss_conf_disc = loss_conf_disc / len(outputs)
	loss_pose_disc = loss_pose_disc / len(outputs)
	# Add discriminator loss
	return {
		'loss': (loss_conf_real + loss_conf_disc + loss_pose_real + loss_pose_disc),
		'conf_disc_real': loss_conf_real,
		'pose_disc_real': loss_pose_real,
		'conf_disc_fake': loss_conf_disc,
		'pose_disc_fake': loss_pose_disc,
	}

#-----------------------gen loss------------conf disc loss------------pose_disc_loss-------------------------------------------------#
def gen_single_loss(ground_truth,
			 outputs,
			 pose_discriminator,
			 mode='mse',
			 alpha=1/220.,
			 beta=1/180.
			):
	'''
	Get generator loss
	'''

	gt_maps = torch.cat([ground_truth['occlusions']], dim=0)   #occlusions    
    
#	loss_recon_0 = 0.0
#	loss_recon_1 = 0.0
	loss_recon   = 0.0
	loss_pose_disc = 0.0
	for output in outputs:
		loss_recon = loss_recon + get_loss_recon(output, gt_maps, mode)
#		loss_recon_1 = loss_recon_1 + get_loss_recon(output[1], gt_maps, mode)
#		loss_recon   = loss_recon + (loss_recon_0 + loss_recon_1)/2.0       
		# loss_pose_disc = loss_pose_disc + get_loss_disc(output, pose_discriminator)

	loss_recon = loss_recon   #/ len(outputs)
	loss_pose_disc = loss_pose_disc / len(outputs)

	# Add discriminator loss
	return {
		'loss': loss_recon, #+ beta*loss_pose_disc,
		'recon': loss_recon,
		'pose_disc': loss_pose_disc,
	}


def disc_single_loss(ground_truth,
			 outputs,
			 pose_discriminator,
			 alpha=1/220.0,
			 beta=1/180.0, 
			 detach=False):
	'''
	Get discriminator loss
	'''

	gt_maps = torch.cat([ground_truth['heatmaps'], ground_truth['heatmaps']], 1)  ##
	loss_pose_real = get_loss_disc_bce_with_logits(gt_maps, pose_discriminator, ground_truth = ground_truth['heatmaps'])


	loss_pose_disc = 0.0
	# TODO: fix the expression in function get_loss_disc for CGAN. Currently, implements traditional GAN
	# False for generator
    
    
	for output in outputs:
		loss_pose_disc = loss_pose_disc + get_loss_disc_bce_with_logits(output, pose_discriminator, detach=detach, real=False )
		#loss_pose_disc = loss_pose_disc + get_loss_disc(output, pose_discriminator, detach=detach, real=False)

	loss_pose_disc = loss_pose_disc / len(outputs)
    
	print("real confidence-discriminator loss: ", format(loss_pose_real))
	print("fake confidence-discriminator loss: ", format(loss_pose_disc))

	# Add discriminator loss
	return {
		'loss': loss_pose_real + loss_pose_disc,
		'pose_disc_real': loss_pose_real,
		'pose_disc_fake': loss_pose_disc,
	}


def disc_loss_pose(ground_truth,
			 outputs,
			 images,
			 pose_discriminator,
			 alpha=1/220.0,
			 beta=1/180.0,
			 detach=False,
			 p_fake = None):
	'''
	Get pose discriminator loss
	'''
	gt_maps = torch.cat([ground_truth['heatmaps']], 1)   ##ground_truth['occlusions']] #, ground_truth['heatmaps']
	gt_w_images = torch.cat([gt_maps, images], 1)

	loss_pose_real = get_loss_disc_bce_with_logits(gt_w_images, pose_discriminator, ground_truth =ground_truth['heatmaps'])
	#loss_pose_real = get_loss_disc(gt_w_images, pose_discriminator)
	# False for generator
#	print("i have reached here ")   
#	print(loss_pose_real)
	loss_pose_disc =0.0
	for output in outputs:
        
		loss_pose_disc = loss_pose_disc + \
			get_loss_disc_bce_with_logits(torch.cat([output, images], 1), pose_discriminator, detach=True, real=False, p_fake = p_fake)
			#get_loss_disc_bce_with_logits(torch.cat([output, images], 1), pose_discriminator, detach=detach, real=False)
        

        
	loss_pose_disc = loss_pose_disc / len(outputs)
	# Add discriminator loss
#	print("real pose-discriminator loss: ", format(loss_pose_real))
#	print("fake pose-discriminator loss: ", format(loss_pose_disc))
	return {
		'loss': (loss_pose_real + loss_pose_disc),
		'pose_disc_real': loss_pose_real,
		'pose_disc_fake': loss_pose_disc,
	}


