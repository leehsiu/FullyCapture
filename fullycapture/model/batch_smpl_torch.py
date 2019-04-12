""" 
Tensorflow Adam implementation as batch.
Specify joint types:
'coco': Returns COCO+ 19 joints
'lsp': Returns H3.6M-LSP 14 joints
Note: To get original smpl joints, use self.J_transformed
"""

import numpy as np
import cPickle as pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from totaldensify.model.batch_rodrigues_torch import *

def kinematric_torch(Rs,Js,parent):
    #Rs NxJx3x3
    #Js NxJx3
    n_batch_ = Rs.shape[0]
    n_J_ = Rs.shape[1]

    # Js NxJx3 -> NxJx3x1
    Js = torch.unsqueeze(Js,-1)

    outT = n_J_*[None]
    outJ = n_J_*[None]


    outT[0] = make_A_torch(n_batch_,Rs[:,0,:,:],Js[:,0])

    for idj in range(1,parent.shape[0]):
        ipar = parent[idj]
        j_here = Js[:,idj] - Js[:,ipar]
        A_here = make_A_torch(n_batch_,Rs[:,idj],j_here)
        outT[idj] = torch.matmul(outT[ipar],A_here)


    res = torch.stack(outT,dim=1)
    new_J = res[:,:,:3,3]

    Js_w0 = torch.cat([Js,torch.zeros(n_batch_,n_J_,1,1,dtype=torch.float32).cuda()],dim=2)
    init_bone = torch.matmul(res,Js_w0)
    #init_bone = F.pad(init_bone,[[0, 0], [0, 0], [0, 0], [3, 0]])
    init_bone = F.pad(init_bone,(3,0,0,0,0,0,0,0))
    A = res - init_bone
    return new_J,A


class SmplModelTorch:
    def __init__(self,pkl_path,reg_type='total'):
        with open(pkl_path, 'r') as f:
            dd = pickle.load(f)
        self.mu_ = dd['v_template']
        self.n_v_ = [self.mu_.shape[0],3]
        self.n_betas_ = dd['shapedirs'].shape[-1]
        self.shapedirs_ = np.reshape(dd['shapedirs'],[-1,self.n_betas_]).T
        self.J_reg_ = dd['J_regressor'].T.todense()
        
        self.kin_parents_ = dd['kintree_table'][0].astype(np.int32)
        self.blendW_ = dd['weights']
        
        if reg_type=='total':
            self.J_reg_coco25_ = dd['J_regressor_total'].T.todense()

        self.n_J_ = self.J_reg_.shape[1]

        #create cuda variable
        self.shapedirs_cu_ = torch.tensor(self.shapedirs_,dtype=torch.float32).cuda()
        self.mu_cu_ = torch.tensor(self.mu_,dtype=torch.float32).cuda()
        self.J_reg_cu_ = torch.tensor(self.J_reg_,dtype=torch.float32).cuda()
        self.J_reg_coco25_cu_ = torch.tensor(self.J_reg_coco25_,dtype=torch.float32).cuda()
        self.blendW_cu_ = torch.tensor(self.blendW_,dtype=torch.float32).cuda()
    

    def __call__(self,betas,theta,reg_type='total'):
        n_batch_ = betas.shape[0]

        #(N x 10) x (10 x V*3) = N x V*3
        v_res = torch.mm(betas,self.shapedirs_cu_)
        #(N x V*3) --> (N x V x 3)

        v_res = v_res.reshape(-1,self.n_v_[0],self.n_v_[1])
        v_shaped = v_res + self.mu_cu_

        #    (NxV) x (VxJ)
        #    (NxV) x (VxJ)
        #    (NxV) x (VxJ)
        Jx = torch.mm(v_shaped[:,:,0],self.J_reg_cu_)
        Jy = torch.mm(v_shaped[:,:,1],self.J_reg_cu_)
        Jz = torch.mm(v_shaped[:,:,2],self.J_reg_cu_)        
        # J as (NxJx3)
        J = torch.stack([Jx,Jy,Jz],dim=2)

        # theta (NxJx3) -> (N*Jx3)
        theta_vec = theta.view(-1,3)
        quat, Rs = batch_rodrigues_torch(theta_vec,transpose_r=False)
        # Rs (N*Jx3x3) -> (NxJx3x3)

        Rs_batch = Rs.view(-1,self.n_J_,3,3)

        J_new_, A = kinematric_torch(Rs_batch,J,self.kin_parents_)

        #Skining
        W_tile_ = self.blendW_cu_.repeat(n_batch_,1)
        W = W_tile_.view(n_batch_,-1,self.n_J_)
        A_vec = A.view(n_batch_,self.n_J_,16)
        T = torch.matmul(W,A_vec).view(n_batch_,-1,4,4)

        v_posed_res = torch.cat([v_shaped,torch.ones(n_batch_,self.n_v_[0],1,dtype=torch.float32).cuda()],dim=2)
        v_homo = torch.matmul(T,torch.unsqueeze(v_posed_res,-1))

        verts = v_homo[:,:,:3,0]

        if reg_type=='legacy':
            J_x_ = torch.matmul(verts[:,:,0],self.J_reg_cu_)
            J_y_ = torch.matmul(verts[:,:,1],self.J_reg_cu_)
            J_z_ = torch.matmul(verts[:,:,2],self.J_reg_cu_)
        elif reg_type=='total':
            J_x_ = torch.matmul(verts[:,:,0],self.J_reg_coco25_cu_)
            J_y_ = torch.matmul(verts[:,:,1],self.J_reg_coco25_cu_)
            J_z_ = torch.matmul(verts[:,:,2],self.J_reg_coco25_cu_)
        else:
            raise ValueError('Known regressor type')
        J_out_ = torch.stack([J_x_,J_y_,J_z_],dim=2)

        return verts, J_out_
    
