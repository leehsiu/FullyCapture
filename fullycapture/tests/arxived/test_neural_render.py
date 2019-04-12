import sys
import torch
import numpy as np
import time
import torch
#import totaldensify.optimizer.torch_smplify as smplify_util
from totaldensify.model.batch_adam_torch import AdamModelTorch
from totaldensify.model.batch_smpl_torch import SmplModelTorch
from totaldensify.model.batch_smpl import SmplModel

#from totaldensify.vis.glut_viewer import glut_viewer
import cPickle as pickle
import totaldensify.data.dataIO as dataIO
import matplotlib.pyplot as plt
import totaldensify.vis.plot_vis as plot_vis

import neural_renderer as nr


#import neural_render as nr
fig, ax = plt.subplots()
def projection(vt,K,R,t,imsize):
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
    vt = torch.stack([x,y,torch.ones_like(z)],dim=-1)
    vt = torch.matmul(vt,K.transpose(1,2))
    u,v = vt[:,:,0],vt[:,:,1]
    u = 2 * (u - imsize/2.) / imsize
    v = 2 * (v - imsize/2.) / imsize

    #normlize vt to [-1,1]

    vt = torch.stack([u,v,z],dim=-1)

    return vt

def test_load_data(img_path,root_path):
    smpl_path = '/home/xiul/workspace/TotalDensify/models/smplModel_with_coco25_reg.pkl'
    smplModelGPU = SmplModelTorch(pkl_path=smpl_path,reg_type='coco25')
    smplModelCPU = SmplModel(pkl_path=smpl_path,reg_type='coco25')


    fit_data = dataIO.prepare_data_total(root_path,img_path)
    # print(fit_data)
    ax.imshow(fit_data['img'])
    # plot_vis.plot_coco25_joints(fit_data['joints'],fit_data['joints_weight'],ax,'r')
    # plt.show()
    betas_init_cu = torch.tensor(fit_data['betas_init'][None,:].astype(np.float64)).cuda()
    pose_init_cu = torch.tensor(fit_data['pose_init'].reshape(24,3)[None,:,:].astype(np.float64)).cuda()
    trans_init_cu = torch.tensor(fit_data['trans_init'][None,:]).cuda()

    vts,jts = smplModelGPU(betas_init_cu,pose_init_cu,reg_type='coco25')
    vts = vts/100.0
    vts += trans_init_cu
    jts += trans_init_cu

    vts.float()
    faces = smplModelCPU.f.astype(np.int32)
    faces_cu = torch.tensor(faces[None,:,:]).cuda()
    
    texture_size = 2

    render = nr.Renderer().cuda()
    textures = torch.ones(1, faces_cu.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    K = np.eye(3)
    R = np.eye(3)
    t = np.zeros((1,3))
    cam = fit_data['cam']
    K[0,0] = cam[0]
    K[1,1] = cam[0]
    K[0,2] = cam[1]
    K[1,2] = cam[2]

    img_size = fit_data['img'].shape[1]
    K_cu = torch.tensor(K[None,:,:]).cuda()
    R_cu = torch.tensor(R[None,:,:]).cuda()
    t_cu = torch.tensor(t[None,:,:]).cuda()
    K_cu = K_cu.float()
    R_cu = R_cu.float()
    t_cu = t_cu.float()

def smplify2d_adam(self,totalModel,j2d_t,init_param,n_iter,reg_type):
 #   vts_float = vts.type(torch.cuda.FloatTensor)
    print(fit_data['img'].shape)
    # vts_proj = projection(vts_float,K_cu,R_cu,t_cu,img_size)
    # vts_proj_2d = vts_proj.detach().cpu().numpy()[0]
    # print(vts_proj_2d.shape)
    # ax.scatter(vts_proj_2d[:,0],vts_proj_2d[:,1])
    # plt.show()
    print(img_size)
    images = render.render(vts_float,faces_cu,textures,K_cu,R_cu,t_cu,None,img_size)
    print(images.shape)
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    print(image.shape)
    img_uint8 = (255*image).astype(np.uint8)
    ax.imshow(img_uint8)
    plt.show()
    #writer.append_data((255*image).astype(np.uint8))

if __name__=='__main__':
    img_path = '/home/xiul/databag/denseFusion/images/run.jpg'
    root_path = '/home/xiul/databag/denseFusion'
    test_load_data(img_path,root_path)