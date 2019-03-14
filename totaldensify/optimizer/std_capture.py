import sys
import json
import math
import time
import numpy as np
import numpy.linalg as nlg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from totaldensify.model.batch_rodrigues_torch import batch_rodrigues_torch
import totaldensify.geometry.rigid_align as align_tools
from totaldensify.model.batch_smpl_torch import SmplModelTorch
from totaldensify.model.batch_smpl import SmplModel
import totaldensify.optimizer.bodyprior as prior_utils
from totaldensify.cpp.totalclib import TotalCLib
from totaldensify.utils.config import cfg
import cv2
from totaldensify.vis.egl_render import EglRender

class StdCapture(object):
    def __init__(self,smpl_path,reg_type):
        self.smpl_path  = smpl_path
        self.reg_type=  reg_type
    
        print('loading models and custom c++ operations')
        self.smplWrapperGPU = SmplModelTorch(pkl_path=self.smpl_path,reg_type=self.reg_type)
        self.smplWrapperCPU = SmplModel(pkl_path=self.smpl_path,reg_type=self.reg_type)
        self.cops = TotalCLib(lib_file='../../build/libtotalCops.so')
        self.cops.load_SmplModel(smpl_path)

        #setup initial fitting weights.
        jtr_w = cfg.CAPTURE.BODY_WEIGHT
        fig_w = cfg.CAPTURE.HAND_WEIGHT
        shape_reg_w = cfg.CAPTURE.SHAPE_REG
        pose_reg_w = cfg.CAPTURE.POSE_REG
        motion_reg_w = cfg.CAPTURE.MOTION_REG


        self.cops.lib.setup_fit_options(jtr_w,fig_w,shape_reg_w,pose_reg_w,motion_reg_w)

        self.J_template = self.smplWrapperCPU.J_template



        print('building pose prior')
        pose_prior = np.loadtxt('../data/kmeans.txt')
        pose_prior = pose_prior.reshape(-1,24,3)
        pose_prior[:,0,:] = 0
        n_prior = pose_prior.shape[0]
        self.prior_pose = np.zeros((n_prior,22+15+15,3),dtype=np.float32)
        self.prior_pose[:,0:22,:] = pose_prior[:,0:22,:]
        self.n_prior = n_prior
        
        zero_shape = np.zeros((n_prior,10))
        _,self.prior_joints = self.smplWrapperCPU(zero_shape,self.prior_pose)


        print('building render')
        self.egl_render = EglRender(1280,720)

        self.model_dct = prior_utils.LinearDCT(100,'dct',norm='ortho').cuda()
        self.model_idct = prior_utils.LinearDCT(100,'idct',norm='ortho').cuda()
        self.l2error_eval = torch.nn.MSELoss(reduction='None').cuda()
        print('Finished Iniitialization')
    def stage0_global_align_torso(self,j3d_t,reg_type='coco25'):
        #0. calculate global rotation translation from torso.
        # Batch is frame-wise
        n_batch = j3d_t.shape[0]
        theta_rot = np.zeros((n_batch,1,3))
        trans = np.zeros((n_batch,1,3))
        for i,c_j3d in enumerate(j3d_t):
            R,t = align_tools.rigid_align_cpu(self.J_template[self.id_torso_coco25,:],c_j3d[self.id_torso_coco25,:])
            vec,_ = cv2.Rodrigues(R)
            theta_rot[i] = vec.T
            trans[i] = t
        return theta_rot,trans
    
    def __align_with_prior_pose(self,j4d_t):
        #* Fixed bug here, some joints of j3d_t is not visible, exclude them by weights.
        #* x,y,z,w * J
        j3d_t = j4d_t[:,:3]
        w, = np.where(j4d_t[:,3]>1e-2)
        if len(w)<=15:
            return np.zeros((3,1)),np.zeros((3,1)),0
        theta_rotation = np.zeros((self.n_prior,3,3),dtype=np.float32)
        trans = np.zeros((self.n_prior,1,3),dtype=np.float32)
        align_error = np.zeros((self.n_prior,1),dtype=np.float32)
        for i,c_prior in enumerate(self.prior_joints):
            
            R,t = align_tools.rigid_align_cpu(c_prior[w,:],j3d_t[w,:])
            #calc_error
            c_err = nlg.norm(np.matmul(c_prior,R)+t-j3d_t)
            theta_rotation[i] = R
            trans[i] = t
            align_error[i] = c_err
        
        best_fit = np.argmin(align_error)

        rot_vec, _ = cv2.Rodrigues(theta_rotation[best_fit])
        trans_v = self.J_template[0,:].dot(theta_rotation[best_fit]) + trans[best_fit]
        return rot_vec,trans_v,best_fit

    def stage0_global_align_full(self,j4d_t,reg_type='coco25'):
        #0. calculate global rotation translation using all joints and the prior
        # Batch is parameter-wise 
        # j4d_t  = x,y,z,w * J
        n_sample = j4d_t.shape[0]
        thetas = np.zeros((n_sample,52,3))
        trans = np.zeros((n_sample,1,3))
        for i,c_j3d in enumerate(j4d_t):
            rot_vec,t,best_fit  = self.__align_with_prior_pose(c_j3d)
            thetas[i] = self.prior_pose[best_fit]
            thetas[i,0,:] = rot_vec.T
            trans[i] = t
            print('model {} finished'.format(i))
        return thetas,trans

    def fit_stage0_lm_ceres(self,j3d_t,j3d_w,init_param):
        #with initial parameters, we fit using simpler
        # fix me later to put the code here.
        #betas_n,theta_n,trans_n = capUtils.cops.smpl_fit_stage1(joints_total[i],betas[i],theta[i]*-1,trans[i],0,True,False)
        return 0


    #not working. 
    def fit_stage1_spatial_temporal(self,j3d_t,j3d_w,v3d_t,v3d_w,init_param,n_iter=100):
        #0. project to dct space
        #dct_init = prior_utils.dct(init_param)
        
        #number of DCT basis length
        n_batch = 100
        assert(j3d_t.shape[0]==n_batch)

        betas_cu_one = torch.tensor(
            init_param['betas'], requires_grad=True, device='cuda',dtype=torch.float32).cuda()
        thetas_base_cu = torch.tensor(init_param['thetas'],dtype=torch.float32,requires_grad=False).cuda()
        thetas_delta_cu = torch.zeros(n_batch,52,3,requires_grad=True,device='cuda',dtype=torch.float32).cuda()

        trans_cu = torch.tensor(
            init_param['trans'],requires_grad=True, device = 'cuda',dtype=torch.float32).cuda()
        
        
        betas_cu = betas_cu_one.repeat(n_batch,1)
        thetas_cu_zeros = torch.zeros_like(thetas_delta_cu,dtype=torch.float32).cuda()
        betas_cu_zeros = torch.zeros_like(betas_cu_one,dtype=torch.float32).cuda()

        weight_coeff = torch.zeros(6890*3,100,dtype=torch.float32).cuda()
        #using different strategy. first try. direct linaer
        for freq in range(20,100):
            weight_coeff[:,freq] = freq*1.0/100
        coeff_cu_zeros = torch.zeros_like(weight_coeff,dtype=torch.float32).cuda()

        j3d_t_cu = torch.tensor(j3d_t,dtype=torch.float32).cuda()
        v3d_t_cu = torch.tensor(v3d_t,dtype=torch.float32).cuda()
        j3d_w_cu = torch.tensor(j3d_w[:,:,None],dtype=torch.float32).cuda()
        v3d_w_cu = torch.tensor(v3d_w,dtype=torch.float32).cuda()
        optimizer = torch.optim.Adam([{'params':betas_cu_one,'lr':1e-1},
                                      {'params':thetas_delta_cu,'lr':1e-1},
                                      {'params':trans_cu}], lr=1e-1)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=25,gamma=0.9)
        l2error_eval = torch.nn.MSELoss(reduction='None').cuda()


        for i in range(n_iter):
            scheduler.step()
            betas_cu = betas_cu_one.repeat(n_batch,1)
            thetas_cu = thetas_base_cu + thetas_delta_cu
            v3d_pred, j3d_pred = self.smplWrapperGPU(betas_cu,thetas_cu)
            v3d_pred = v3d_pred + trans_cu
            j3d_pred = j3d_pred + trans_cu

            dct_coeff = self.model_dct(v3d_pred.view(-1,6890*3).transpose(1,0))

            resi_j3d = l2error_eval(j3d_pred,j3d_t_cu)*j3d_w_cu/100.0
            resi_v3d = l2error_eval(v3d_pred,v3d_t_cu)*v3d_w_cu/100.0
            resi_beta = l2error_eval(betas_cu_one,betas_cu_zeros)
            resi_theta = l2error_eval(thetas_delta_cu,thetas_cu_zeros)
            resi_dct = l2error_eval(dct_coeff,coeff_cu_zeros)*weight_coeff

            resi_total = 1*torch.sum(resi_j3d) + 1e-1*torch.sum(resi_v3d) + 1e1*torch.sum(resi_beta) + 1e1*torch.sum(resi_theta) + 1e-2*torch.sum(resi_dct)

            optimizer.zero_grad()	
            resi_total.backward()
            optimizer.step()
            print('loss in iter [{}/{}] : {}'.format(i,n_iter,resi_total.item()))

        #Evaluator
        return betas_cu.detach().cpu().numpy(),thetas_cu.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()   

    def fit_stage2_photometric(self,j3d_t,image_t,cam_t,init_param,n_iter=100,reg_type='coco25'):
        #0. get visibility map
        #assign different color the the vertex, render and read back to get the.
        n_batch = j3d_t.shape[0]
        betas_cu_one = torch.tensor(
            init_param['betas'], requires_grad=True, device='cuda',dtype=torch.float32).cuda()
        
        thetas_base_cu = torch.tensor(init_param['thetas'],dtype=torch.float32,requires_grad=False).cuda()
        thetas_delta_cu = torch.zeros(n_batch,52,3,requires_grad=True,device='cuda',dtype=torch.float32).cuda()
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

        for i in range(n_iter):
            thetas_cu = 1

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
            
           
