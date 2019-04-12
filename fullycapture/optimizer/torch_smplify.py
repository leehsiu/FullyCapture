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

class BodyRecon:
    def __init__(self):
        self.mseLoss = torch.nn.MSELoss(reduction='sum').cuda()
        id_torso_coco25 = [2, 5, 9, 12]
        weight_coco25 = np.ones((1, 25, 3))
        weight_coco25[:, id_torso_coco25, :] =  3.0
        
        weight_coco25[:,20:,:] = 0 #No foot currently


        self.weight_coco25_cu = torch.tensor(weight_coco25,dtype=torch.float32).cuda()

    def smplify3d_sgd_mannual(self, totalModel, j3d_t, init_param, n_iter, reg_type):
        n_batch = j3d_t.shape[0]
        betas_cu = torch.tensor(
            init_param['betas'], requires_grad=True, device='cuda').cuda()
        thetas_cu = torch.tensor(
            init_param['thetas'], requires_grad=True, device='cuda').cuda()
        j3d_t_cu = torch.tensor(j3d_t).cuda()
        j3d_t_cu = j3d_t_cu.permute(0, 2, 1)

        mask_torso_cu = self.mask_torso_coco25_cu.repeat(n_batch, 1, 1)

        assert(reg_type == 'coco25')
        # optimizer = torch.optim.Adam(
        # 	[{'params':betas_cu},{'params':}, lr=0.1).cuda()
        # #0. fit torso only

        for i in range(n_iter):
            _, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred.permute(0, 2, 1)
            loss = self.mseLoss(j3d_t_cu, j3d_pred)
            loss.backward()
            betas_cu.data -= 1e-1 * betas_cu.grad.data
            thetas_cu.data -= 1e-1 * thetas_cu.grad.data
            betas_cu.grad.zero_()
            thetas_cu.grad.zero_()
            print('loss in iter [{}/{}] : {}'.format(i, n_iter, loss.item()))

        return betas_cu.detach().cpu().numpy(), thetas_cu.detach().cpu().numpy()

    #def __projection(self,vt,K,R,t,imsize):
    def smplify2d_adam(self,totalModel,j2d_t,init_param,n_iter,reg_type):

        n_batch = j2d_t.shape[0]
        betas_cu = torch.tensor(
            init_param['betas'],requires_grad=True, device='cuda').cuda()
        thetas_cu = torch.tensor(
            init_param['thetas'],requires_grad=True, device='cuda').cuda()
        trans_cu = torch.tensor(
            init_param['trans'],requires_grad=True, device = 'cuda').cuda()
        cam_K_cu = torch.tensor(init_param['K']).cuda()
        cam_R_cu = torch.tensor(init_param['R']).cuda()
        cam_t_cu = torch.tensor(init_param['t']).cuda()

        thetas_cu_zeros = torch.zeros_like(thetas_cu).cuda()
        betas_cu_zeros = torch.zeros_like(betas_cu).cuda()

        weight_cu = self.weight_coco25_cu.repeat(n_batch, 1, 1)


        j3d_t_cu = torch.tensor(j3d_t).cuda()

        optimizer = torch.optim.Adam([{'params':betas_cu},
                                      {'params':thetas_cu},
                                      {'params':trans_cu}], lr=0.5)
        # #0. fit torso only
        for i in range(n_iter):
            _, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred + trans_cu
            j3d_pred = j3d_pred * weight_cu
            j3d_t_cu_n = j3d_t_cu * weight_cu

            loss = self.mseLoss(j3d_t_cu_n, j3d_pred)
            loss_norm = self.mseLoss(thetas_cu,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu,betas_cu_zeros)
            loss_total = 0.1*loss + 10.0*loss_norm + 1*loss_beta
            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

        for i in range(n_iter):
            _, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred + trans_cu
            j3d_pred = j3d_pred * weight_cu
            j3d_t_cu_n = j3d_t_cu * weight_cu

            loss = self.mseLoss(j3d_t_cu_n, j3d_pred)
            loss_norm = self.mseLoss(thetas_cu,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu,betas_cu_zeros)
            loss_total = 0.1*loss + 1*loss_norm + 1*loss_beta
            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

        for i in range(n_iter):
            _, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred + trans_cu
            j3d_pred = j3d_pred * weight_cu
            j3d_t_cu_n = j3d_t_cu * weight_cu

            loss = self.mseLoss(j3d_t_cu_n, j3d_pred)
            loss_norm = self.mseLoss(thetas_cu,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu,betas_cu_zeros)
            loss_total = 0.1*loss + 0.1*loss_norm + 1*loss_beta
            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

        return betas_cu.detach().cpu().numpy(),thetas_cu.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()



    def densify3d_adam(self,totalModel,j3d_t,v3d_t,init_param,joint_mask,verts_mask,n_iter=100,reg_type='coco25'):
        n_batch = j3d_t.shape[0]
        betas_cu_one = torch.tensor(
            init_param['betas'],requires_grad=True, device='cuda',dtype=torch.float32).cuda()
        thetas_cu = torch.tensor(
            init_param['thetas'],requires_grad=True, device='cuda',dtype=torch.float32).cuda()
        trans_cu = torch.tensor(
            init_param['trans'],requires_grad=True, device = 'cuda',dtype=torch.float32).cuda()
        
        betas_cu = betas_cu_one.repeat(n_batch,1)
        
        thetas_cu_zeros = torch.zeros_like(thetas_cu,dtype=torch.float32,requires_grad=False).cuda()
        betas_cu_zeros = torch.zeros_like(betas_cu_one,dtype=torch.float32,requires_grad=False).cuda()
        thetas_weight = torch.ones_like(thetas_cu,dtype=torch.float32,requires_grad=False).cuda()
        thetas_weight[:,0,:] = 0

        v3d_weight_cu = torch.tensor(verts_mask,dtype=torch.float32).cuda()

        
        weight_cu = self.weight_coco25_cu.repeat(n_batch, 1, 1)

        j3d_t_cu = torch.tensor(j3d_t,dtype=torch.float32).cuda()
        v3d_t_cu = torch.tensor(v3d_t,dtype=torch.float32).cuda()

        optimizer = torch.optim.Adam([{'params':betas_cu_one,'lr':1e-1},
                                      {'params':thetas_cu},
                                      {'params':trans_cu}], lr=1)
    
        # #0. fit torso only
        for i in range(n_iter):
            betas_cu = betas_cu_one.repeat(n_batch,1)
            v3d_pred, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred + trans_cu
            v3d_pred = v3d_pred + trans_cu

            j3d_pred = j3d_pred * weight_cu /100.0
            j3d_t_cu_tmp = j3d_t_cu * weight_cu /100.0
            v3d_pred = v3d_pred * v3d_weight_cu /100.0
            v3d_t_cu_tmp = v3d_t_cu * v3d_weight_cu /100.0

            loss_j3d = self.mseLoss(j3d_pred,j3d_t_cu_tmp)
            loss_v3d = self.mseLoss(v3d_pred,v3d_t_cu_tmp)

            loss_norm = self.mseLoss(thetas_cu*thetas_weight,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu_one,betas_cu_zeros)

            loss_total = loss_j3d + 0.001*loss_v3d + 1*loss_norm + loss_beta
            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

        for i in range(n_iter):
            betas_cu = betas_cu_one.repeat(n_batch,1)
            v3d_pred, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred + trans_cu
            v3d_pred = v3d_pred + trans_cu

            j3d_pred = j3d_pred * weight_cu /100.0
            j3d_t_cu_tmp = j3d_t_cu * weight_cu /100.0
            v3d_pred = v3d_pred * v3d_weight_cu /100.0 
            v3d_t_cu_tmp = v3d_t_cu * v3d_weight_cu /100.0

            loss_j3d = self.mseLoss(j3d_pred,j3d_t_cu_tmp)
            loss_v3d = self.mseLoss(v3d_pred,v3d_t_cu_tmp)

            loss_norm = self.mseLoss(thetas_cu*thetas_weight,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu_one,betas_cu_zeros)

            loss_total = loss_j3d + 0.001*loss_v3d + 1e-2*loss_norm + loss_beta
            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

        for i in range(n_iter):
            betas_cu = betas_cu_one.repeat(n_batch,1)
            v3d_pred, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred + trans_cu
            v3d_pred = v3d_pred + trans_cu

            j3d_pred = j3d_pred * weight_cu /100.0 
            j3d_t_cu_tmp = j3d_t_cu * weight_cu /100.0 
            v3d_pred = v3d_pred * v3d_weight_cu /100.0 
            v3d_t_cu_tmp = v3d_t_cu * v3d_weight_cu /100.0

            loss_j3d = self.mseLoss(j3d_pred,j3d_t_cu_tmp)
            loss_v3d = self.mseLoss(v3d_pred,v3d_t_cu_tmp)

            loss_norm = self.mseLoss(thetas_cu*thetas_weight,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu_one,betas_cu_zeros)

            loss_total = loss_j3d + 0.001*loss_v3d + 1e-4*loss_norm +1e-1*loss_beta
            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))
        return betas_cu.detach().cpu().numpy(),thetas_cu.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()
    
    def smplify3d_adam(self,totalModel,j3d_t,init_param,joint_mask=None,n_iter=100,reg_type='coco25'):
        n_batch = j3d_t.shape[0]
        betas_cu = torch.tensor(
            init_param['betas'], requires_grad=True, device='cuda',dtype=torch.float32).cuda()
        thetas_cu = torch.tensor(
            init_param['thetas'], requires_grad=True, device='cuda',dtype=torch.float32).cuda()
        trans_cu = torch.tensor(
            init_param['trans'],requires_grad=True, device = 'cuda',dtype=torch.float32).cuda()
        thetas_cu_zeros = torch.zeros_like(thetas_cu).cuda()
        betas_cu_zeros = torch.zeros_like(betas_cu).cuda()

        if joint_mask is None:
            weight_cu = self.weight_coco25_cu.repeat(n_batch, 1, 1)
        else:
            weight_cu = torch.tensor(joint_mask,dtype=torch.float32).cuda()
            
        j3d_t_cu = torch.tensor(j3d_t,dtype=torch.float32).cuda()

        optimizer = torch.optim.Adam([{'params':betas_cu},
                                      {'params':thetas_cu},
                                      {'params':trans_cu}], lr=1)
        # #0. fit torso only
        for i in range(n_iter):
            _, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred + trans_cu
            j3d_pred = j3d_pred * weight_cu /100.0
            j3d_t_cu_n = j3d_t_cu * weight_cu /100.0
            loss = self.mseLoss(j3d_t_cu_n, j3d_pred)
            loss_norm = self.mseLoss(thetas_cu,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu,betas_cu_zeros)
            loss_total = loss + 1*loss_norm + loss_beta
            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

        for i in range(n_iter):
            _, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred + trans_cu
            j3d_pred = j3d_pred * weight_cu /100.0
            j3d_t_cu_n = j3d_t_cu * weight_cu /100.0

            loss = self.mseLoss(j3d_t_cu_n, j3d_pred)
            loss_norm = self.mseLoss(thetas_cu,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu,betas_cu_zeros)
            loss_total = loss + 1e-2*loss_norm + 1*loss_beta
            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

        for i in range(n_iter):
            _, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
            j3d_pred = j3d_pred + trans_cu
            j3d_pred = j3d_pred * weight_cu /100.0
            j3d_t_cu_n = j3d_t_cu * weight_cu /100.0

            loss = self.mseLoss(j3d_t_cu_n, j3d_pred)
            loss_norm = self.mseLoss(thetas_cu,thetas_cu_zeros)
            loss_beta = self.mseLoss(betas_cu,betas_cu_zeros)
            loss_total = loss + 1e-4*loss_norm + 1*loss_beta
            optimizer.zero_grad()	
            loss_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

        return betas_cu.detach().cpu().numpy(),thetas_cu.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()
    #def __body_prior(self,)
    #internal function
    



    def __projection(self,vt,K,R,t,imsize):
        '''
        Input
            v: NxVx3 vertices
            K: Nx3x3 intrinsic 
            R: Nx3x3 camera rotation
            t: Nx3x1 camera translation
            imsize: Nx1 imgsize
        Output
            [u,v,z] in image
        '''
        eps = 1e-9
        #v = torch.matmul()
        vt = torch.matmul(vt,R.transpose(2,1)) + t
        x,y,z = vt[:,:,0],vt[:,:,1],vt[:,:,2]
        x_ = x / (z + eps)
        y_ = y / (z + eps)
        #no distortion
        vt = torch.stack([x_,y_,torch.ones_like(z)],dim=-1)
        vt = torch.matmul(vt,K.transpose(1,2))
        u,v = vt[:,:,0],vt[:,:,1]
        u = 2 * (u - imsize/2.) / imsize
        v = 2 * (v - imsize/2.) / imsize

        #normlize vt to [-1,1]

        vt = torch.stack([u,v,z],dim=-1)

        return vt




