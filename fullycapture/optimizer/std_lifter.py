import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from totaldensify.model.batch_adam_torch import AdamModelTorch

from totaldensify.vis.glut_viewer import glut_viewer
import totaldensify.optimizer.bodyprior as prior_utils

#the standard capture pipeline

class StdCapture(object):
    def __init__(self):
        self.mseLoss = torch.nn.MSELoss(reduction='sum').cuda()
        id_torso_coco25 = [2, 5, 8, 9, 12]
        weight_coco25 = np.ones((1, 25, 3)) 
        weight_coco25[:, id_torso_coco25, :] =1.0
        weight_coco25[:, 8, :] =  0.0
        weight_coco25[:, 9, :] =  0.0
        weight_coco25[:, 12, :] =  0.0
        weight_coco25[:, 11, :] =  1.5
        weight_coco25[:, 14, :] =  1.5
        #weight_coco25[:, 12, :] =  0.0
        weight_coco25[:,20:,:] = 0 #No foot currently
        self.weight_coco25_cu = torch.tensor(weight_coco25,dtype=torch.float32).cuda()

        weight_torso = np.zeros((1,25,3))
        weight_torso[:,id_torso_coco25,:] = 1
        self.weight_torso_cu = torch.tensor(weight_torso,dtype=torch.float32).cuda()


        pose_prior = np.loadtxt('../data/kmeans.txt')
        pose_prior = pose_prior.reshape(-1,24,3)
        pose_prior[:,0,:] = 0
        self.pose_prior_init = pose_prior
        self.weight_pose = torch.ones(1,24,3,dtype=torch.float32).cuda()

        self.model_dct = prior_utils.LinearDCT(100,'dct',norm='ortho').cuda()

        self.model_idct = prior_utils.LinearDCT(100,'idct',norm='ortho').cuda()


    def fit_stage0_global_align(self,totalModel,j3d_t,init_param,n_iter=100,reg_type='coco25'):

        #0 calculate global rotation translation from torso.

        n_batch = 128
        j3d_t = j3d_t.repeat(n_batch,axis=0)

        betas_cu_one = torch.tensor(
            init_param['betas'], requires_grad=True, device='cuda',dtype=torch.float32).cuda()
        thetas_base_cu = torch.tensor(
            self.pose_prior_init,dtype=torch.float32).cuda()
        thetas_delta_cu = torch.zeros(n_batch,24,3,requires_grad=True,device='cuda',dtype=torch.float32).cuda()

        trans_cu_one = torch.tensor(
            init_param['trans'],requires_grad=True, device = 'cuda',dtype=torch.float32).cuda()


        betas_cu = betas_cu_one.repeat(n_batch,1)
        trans_cu = trans_cu_one.repeat(n_batch,1,1)
        thetas_cu = thetas_base_cu + thetas_delta_cu
        thetas_cu_zeros = torch.zeros_like(thetas_delta_cu,dtype=torch.float32).cuda()
        betas_cu_zeros = torch.zeros_like(betas_cu_one,dtype=torch.float32).cuda()
        
        weight_cu = self.weight_coco25_cu.repeat(n_batch, 1, 1)
        weight_pose_cu = self.weight_pose.repeat(n_batch,1,1)

        j3d_t_cu = torch.tensor(j3d_t,dtype=torch.float32).cuda()

        optimizer = torch.optim.Adam([{'params':betas_cu_one,'lr':1e-1},
                                      {'params':thetas_delta_cu},
                                      {'params':trans_cu_one}], lr=1)
    

        # #0. fit torso only
        for i in range(n_iter):
            betas_cu = betas_cu_one.repeat(n_batch,1)
            trans_cu = trans_cu_one.repeat(n_batch,1,1)
            thetas_cu = thetas_base_cu + thetas_delta_cu
            _, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred + trans_cu

            j3d_pred = j3d_pred * weight_cu /100.0
            j3d_t_cu_tmp = j3d_t_cu * weight_cu /100.0

            loss_j3d = self.mseLoss(j3d_pred,j3d_t_cu_tmp)
            loss_norm = self.mseLoss(thetas_delta_cu*weight_pose_cu,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu_one,betas_cu_zeros)

            loss_total = loss_j3d + 1e1*loss_norm + loss_beta

            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))
        
        #Evaluator
        return betas_cu.detach().cpu().numpy(),thetas_cu.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()


    def fit_stage2_spatial_temporal(self,totalModel,j3d_t,v3d_t,init_param,joint_mask,verts_mask,n_iter=100,reg_type='coco25'):
        #0. project to dct space
        #dct_init = prior_utils.dct(init_param)
        n_batch = 100
        assert(j3d_t.shape[0]==n_batch)

        betas_cu_one = torch.tensor(
            init_param['betas'], requires_grad=True, device='cuda',dtype=torch.float32).cuda()
        
        thetas_base_cu = torch.tensor(init_param['thetas'],dtype=torch.float32,requires_grad=False).cuda()
        thetas_delta_cu = torch.zeros(n_batch,24,3,requires_grad=True,device='cuda',dtype=torch.float32).cuda()
        trans_cu = torch.tensor(
            init_param['trans'],requires_grad=True, device = 'cuda',dtype=torch.float32).cuda()
        betas_cu = betas_cu_one.repeat(n_batch,1)

        thetas_cu_zeros = torch.zeros_like(thetas_delta_cu,dtype=torch.float32).cuda()
        betas_cu_zeros = torch.zeros_like(betas_cu_one,dtype=torch.float32).cuda()

        weight_cu = self.weight_coco25_cu.repeat(n_batch, 1, 1)
        weight_coeff = torch.zeros(24*3,100,dtype=torch.float32).cuda()
        weight_coeff[:,50:] = 1
        coeff_cu_zeros = torch.zeros_like(weight_coeff,dtype=torch.float32).cuda()
        weight_pose_cu = self.weight_pose.repeat(n_batch,1,1)
        j3d_t_cu = torch.tensor(j3d_t,dtype=torch.float32).cuda()
        v3d_t_cu = torch.tensor(v3d_t,dtype=torch.float32).cuda()

        optimizer = torch.optim.Adam([{'params':betas_cu_one,'lr':1e-1},
                                      {'params':thetas_delta_cu},
                                      {'params':trans_cu}], lr=1e-2)


        v3d_weight_cu = torch.tensor(verts_mask,dtype=torch.float32).cuda()
        # #0. fit torso only
        v3d_t_cu_tmp = v3d_t_cu * v3d_weight_cu /100.0

        for i in range(n_iter*2):
            betas_cu = betas_cu_one.repeat(n_batch,1)
            thetas_cu = thetas_base_cu + thetas_delta_cu
            thetas_cu = thetas_cu.view(-1,24*3).transpose(1,0)
            thetas_cu_coeff = self.model_dct(thetas_cu)
            #print(thetas_cu_coeff[:,0])
            thetas_cu_recon = self.model_idct(thetas_cu_coeff)
            #print(thetas_cu_recon.shape)
            thetas_cu_model = thetas_cu_recon.transpose(1,0).view(-1,24,3).contiguous()
            #print(thetas_cu_model.shape)
            v3d_pred, j3d_pred = totalModel(betas_cu, thetas_cu_model, reg_type)
            j3d_pred = j3d_pred + trans_cu
            v3d_pred = v3d_pred + trans_cu
            j3d_pred = j3d_pred * weight_cu /100.0
            j3d_t_cu_tmp = j3d_t_cu * weight_cu /100.0

            v3d_pred = v3d_pred * v3d_weight_cu /100.0

            loss_j3d = self.mseLoss(j3d_pred,j3d_t_cu_tmp)
            loss_v3d = self.mseLoss(v3d_pred,v3d_t_cu_tmp)

            loss_norm = self.mseLoss(thetas_delta_cu*weight_pose_cu,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu_one,betas_cu_zeros)
            loss_coeff = self.mseLoss(thetas_cu_coeff*weight_coeff,coeff_cu_zeros)

            loss_total = loss_j3d + loss_norm + 1e-2*loss_beta + 1e2*loss_coeff + 1e-1*loss_v3d
            #loss_total = loss_j3d + loss_norm + loss_coeff 

            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))
            print('loss j3d {}'.format(loss_j3d.item()))
            print('loss norm {}'.format(loss_norm.item()))
            print('loss coeff {}'.format(loss_coeff.item()))
            print('loss betas {}'.format(loss_beta.item()))
            print('loss v3d {}'.format(loss_v3d.item()))
        #Evaluator
        return betas_cu.detach().cpu().numpy(),thetas_cu_model.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()   

    def fit_stage2_photometric(self,totalModel,rendered,tar_images,init_param,n_iter=100,reg_type='coco25'):
        #0. project to dct space
        #dct_init = prior_utils.dct(init_param)
        n_batch = 100
        assert(j3d_t.shape[0]==n_batch)

        betas_cu_one = torch.tensor(
            init_param['betas'], requires_grad=True, device='cuda',dtype=torch.float32).cuda()
        
        thetas_base_cu = torch.tensor(init_param['thetas'],dtype=torch.float32,requires_grad=False).cuda()
        thetas_delta_cu = torch.zeros(n_batch,24,3,requires_grad=True,device='cuda',dtype=torch.float32).cuda()
        trans_cu = torch.tensor(
            init_param['trans'],requires_grad=True, device = 'cuda',dtype=torch.float32).cuda()
        betas_cu = betas_cu_one.repeat(n_batch,1)

        thetas_cu_zeros = torch.zeros_like(thetas_delta_cu,dtype=torch.float32).cuda()
        betas_cu_zeros = torch.zeros_like(betas_cu_one,dtype=torch.float32).cuda()

        weight_cu = self.weight_coco25_cu.repeat(n_batch, 1, 1)
        weight_coeff = torch.zeros(24*3,100,dtype=torch.float32).cuda()
        weight_coeff[:,50:] = 1
        coeff_cu_zeros = torch.zeros_like(weight_coeff,dtype=torch.float32).cuda()
        weight_pose_cu = self.weight_pose.repeat(n_batch,1,1)
        j3d_t_cu = torch.tensor(j3d_t,dtype=torch.float32).cuda()
        v3d_t_cu = torch.tensor(v3d_t,dtype=torch.float32).cuda()

        optimizer = torch.optim.Adam([{'params':betas_cu_one,'lr':1e-1},
                                      {'params':thetas_delta_cu},
                                      {'params':trans_cu}], lr=1e-2)


        v3d_weight_cu = torch.tensor(verts_mask,dtype=torch.float32).cuda()
        # #0. fit torso only
        v3d_t_cu_tmp = v3d_t_cu * v3d_weight_cu /100.0

        for i in range(n_iter*2):
            betas_cu = betas_cu_one.repeat(n_batch,1)
            thetas_cu = thetas_base_cu + thetas_delta_cu
            thetas_cu = thetas_cu.view(-1,24*3).transpose(1,0)
            thetas_cu_coeff = self.model_dct(thetas_cu)
            #print(thetas_cu_coeff[:,0])
            thetas_cu_recon = self.model_idct(thetas_cu_coeff)
            #print(thetas_cu_recon.shape)
            thetas_cu_model = thetas_cu_recon.transpose(1,0).view(-1,24,3).contiguous()
            #print(thetas_cu_model.shape)
            v3d_pred, j3d_pred = totalModel(betas_cu, thetas_cu_model, reg_type)
            j3d_pred = j3d_pred + trans_cu
            v3d_pred = v3d_pred + trans_cu
            j3d_pred = j3d_pred * weight_cu /100.0
            j3d_t_cu_tmp = j3d_t_cu * weight_cu /100.0

            v3d_pred = v3d_pred * v3d_weight_cu /100.0

            loss_j3d = self.mseLoss(j3d_pred,j3d_t_cu_tmp)
            loss_v3d = self.mseLoss(v3d_pred,v3d_t_cu_tmp)

            loss_norm = self.mseLoss(thetas_delta_cu*weight_pose_cu,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu_one,betas_cu_zeros)
            loss_coeff = self.mseLoss(thetas_cu_coeff*weight_coeff,coeff_cu_zeros)

            loss_total = loss_j3d + loss_norm + 1e-2*loss_beta + 1e2*loss_coeff + 1e-1*loss_v3d
            #loss_total = loss_j3d + loss_norm + loss_coeff 

            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))
            print('loss j3d {}'.format(loss_j3d.item()))
            print('loss norm {}'.format(loss_norm.item()))
            print('loss coeff {}'.format(loss_coeff.item()))
            print('loss betas {}'.format(loss_beta.item()))
            print('loss v3d {}'.format(loss_v3d.item()))
        #Evaluator
        return betas_cu.detach().cpu().numpy(),thetas_cu_model.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()   
            
           
