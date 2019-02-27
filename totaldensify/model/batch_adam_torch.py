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
#batch

def batch_rodrigues_torch(theta, transpose_r = False):
    #theta N*J x 3
    # batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    
    # quat N*Jx4
    return quat, quat2mat_torch(quat, transpose_r)

def quat2mat_torch(quat, transpose_r = False):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [N*J, 4] <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [N*J, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    if not transpose_r:
        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                              2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                              2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    else:
        rotMat = torch.stack([w2 + x2 - y2 - z2,    2*wz + 2*xy,    2*xz - 2*wy,
                            2*xy - 2*wz,   w2 - x2 + y2 - z2,  2*wx + 2*yz,
                            2*wy + 2*xz,    2*yz - 2*wx,    w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)

    return rotMat
# def make_A_torch(R, t):
#     R_homo = F.pad([R, [[0, 0], [0, 1], [0, 0]]], dim = 0)
#     t_homo = torch.cat([t.view(3,1), torch.ones(1, 1).cuda()], dim = 0)
#     return torch.cat([R_homo, t_homo], dim=1)


#                 R_homo = tf.pad(R, [[0, 0], [0, 1], [0, 0]])
#                 t_homo = tf.concat([t, tf.ones([N, 1, 1])], 1)
#                 return tf.concat([R_homo, t_homo], 2)

def make_A_torch(N,R,t):

    R_homo = torch.cat([R,torch.zeros(N,1,3,dtype=torch.double).cuda()],dim=1)
    t_homo = torch.cat([t.view(N,3,1), torch.ones(N,1,1,dtype=torch.double).cuda()], dim = 1)
    return torch.cat([R_homo, t_homo], dim=2)

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

    Js_w0 = torch.cat([Js,torch.zeros(n_batch_,n_J_,1,1,dtype=torch.double).cuda()],dim=2)
    init_bone = torch.matmul(res,Js_w0)
    #init_bone = F.pad(init_bone,[[0, 0], [0, 0], [0, 0], [3, 0]])
    init_bone = F.pad(init_bone,(3,0,0,0,0,0,0,0))
    A = res - init_bone
    return new_J,A


class AdamModelTorch:
    def __init__(self,pkl_path,reg_type='legacy'):
        with open(pkl_path, 'r') as f:
            dd = pickle.load(f)   
        self.mu_ = dd['mu']
        self.n_v_ = [self.mu_.shape[0],3]
        self.n_betas_ = dd['shapedirs'].shape[-1]
        self.shapedirs_ = np.reshape(dd['shapedirs'],[-1,self.n_betas_]).T
        self.J_reg_ = dd['J_regressor'].T.todense()
        
        self.kin_parents_ = dd['kintree_table'][0].astype(np.int32)
        self.blendW_ = dd['weights']
        
        if reg_type=='coco25':
            self.J_reg_coco25_ = dd['J_regressor_coco25'].T.todense()

        self.n_J_ = self.J_reg_.shape[1]

        #create cuda variable
        self.shapedirs_cu_ = torch.tensor(self.shapedirs_).cuda()
        self.mu_cu_ = torch.tensor(self.mu_).cuda()
        self.J_reg_cu_ = torch.tensor(self.J_reg_).cuda()
        self.J_reg_coco25_cu_ = torch.tensor(self.J_reg_coco25_).cuda()
        self.blendW_cu_ = torch.tensor(self.blendW_).cuda()
    
    def __call__(self,betas,theta,reg_type='legacy'):
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

        v_posed_res = torch.cat([v_shaped,torch.ones(n_batch_,self.n_v_[0],1,dtype=torch.double).cuda()],dim=2)
        v_homo = torch.matmul(T,torch.unsqueeze(v_posed_res,-1))

        verts = v_homo[:,:,:3,0]

        if reg_type=='legacy':
            J_x_ = torch.matmul(verts[:,:,0],self.J_reg_cu_)
            J_y_ = torch.matmul(verts[:,:,1],self.J_reg_cu_)
            J_z_ = torch.matmul(verts[:,:,2],self.J_reg_cu_)
        elif reg_type=='coco25':
            J_x_ = torch.matmul(verts[:,:,0],self.J_reg_coco25_cu_)
            J_y_ = torch.matmul(verts[:,:,1],self.J_reg_coco25_cu_)
            J_z_ = torch.matmul(verts[:,:,2],self.J_reg_coco25_cu_)
        else:
            raise ValueError('Known regressor type')
        J_out_ = torch.stack([J_x_,J_y_,J_z_],dim=2)

        return verts, J_out_
    
