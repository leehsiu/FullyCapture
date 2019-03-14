import sys
import torch
import torch.nn
import numpy as np
import time
#import totaldensify.optimizer.torch_smplify as smplify_util
from totaldensify.model.batch_smpl_torch import SmplModelTorch
from totaldensify.model.batch_smpl import SmplModel
#from totaldensify.vis.glut_viewer import glut_viewer
import totaldensify.optimizer.bodyprior as prior_utils
import cPickle as pickle
import totaldensify.data.dataIO as dataIO
import matplotlib.pyplot as plt
import totaldensify.vis.plot_vis as plot_vis
import neural_renderer as nr

from totaldensify.vis.glut_viewer import glut_viewer
c_map = plt.get_cmap('hsv')
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
    vt = torch.stack([x_,y_,torch.ones_like(z)],dim=-1)
    vt = torch.matmul(vt,K.transpose(1,2))
    u,v = vt[:,:,0],vt[:,:,1]
    u = 2 * (u - imsize/2.) / imsize
    v = 2 * (v - imsize/2.) / imsize

    #normlize vt to [-1,1]

    vt = torch.stack([u,v,z],dim=-1)

    return vt

def densify2d_adam(totalModel,j2d_t,j2d_w,v2d_t,v2d_w,init_param,n_iter,reg_type):


    n_batch = j2d_t.shape[0]
    thetas_delta_cu = torch.zeros(n_batch,24,3,requires_grad=True,device='cuda',dtype=torch.float32).cuda()

    betas_cu_one = torch.tensor(
        init_param['betas'][0][None,:],requires_grad=True, device='cuda').cuda()
    trans_cu = torch.tensor(
        init_param['trans'],requires_grad=True, device = 'cuda').cuda()
    thetas_base_cu = torch.tensor(
        init_param['theta'],device='cuda').cuda()

    betas_cu = betas_cu_one.repeat(n_batch,1)

    cam_K_cu = torch.tensor(init_param['K'],dtype=torch.float32).cuda()
    cam_R_cu = torch.tensor(init_param['R'],dtype=torch.float32).cuda()
    cam_t_cu = torch.tensor(init_param['t'],dtype=torch.float32).cuda()
    j2d_w = torch.tensor(j2d_w,dtype=torch.float32).cuda()
    v2d_w = torch.tensor(v2d_w,dtype=torch.float32).cuda()

    img_size = init_param['img_size']


    thetas_cu = thetas_base_cu + thetas_delta_cu
    thetas_cu_zeros = torch.zeros_like(thetas_delta_cu).cuda()
    betas_cu_zeros = torch.zeros_like(betas_cu).cuda()

    #weight_cu = self.weight_coco25_cu.repeat(n_batch, 1, 1)

    j2d_t = torch.tensor(j2d_t,dtype=torch.float32).cuda() 
    v2d_t = torch.tensor(v2d_t,dtype=torch.float32).cuda()
    
    optimizer = torch.optim.Adam([{'params':betas_cu_one,'lr':1e-1},
                                    {'params':thetas_delta_cu},
                                    {'params':trans_cu}], lr=1e-1)
    l2loss = torch.nn.MSELoss(reduction='sum').cuda()
    #bceloss = torch.nn.BCELoss().cuda()
    dct_criterion = prior_utils.LinearDCT(100,'dct','ortho').cuda()

    weight_coeff = torch.zeros(6890*3,100,dtype=torch.float32).cuda()
    weight_coeff[:,80:] = 1
    coeff_cu_zeros = torch.zeros_like(weight_coeff,dtype=torch.float32).cuda()
        
    for i in range(n_iter):
        betas_cu = betas_cu_one.repeat(n_batch,1)
        thetas_cu = thetas_delta_cu+thetas_base_cu 
        v3d_pred, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
        j3d_pred = j3d_pred /100.0 + trans_cu
        v3d_pred = v3d_pred /100.0 + trans_cu
        v3d_pred_vec = v3d_pred.view(-1,6890*3).transpose(1,0)
        dct_coef = dct_criterion(v3d_pred_vec)



        #j3d_pred = j3d_pred / 100.0
        #images = rendered(vts_float,model_faces_cu,tex,None,K_cu,R_cu,t_cu,None,img_size)
        j2d_pred = projection(j3d_pred,cam_K_cu,cam_R_cu,cam_t_cu,img_size)[:,:,:2]
        v2d_pred = projection(v3d_pred,cam_K_cu,cam_R_cu,cam_t_cu,img_size)[:,:,:2]

        j2d_loss = l2loss(j2d_pred*j2d_w, j2d_t*j2d_w)
        v2d_loss = l2loss(v2d_pred*v2d_w, v2d_t*v2d_w)

        loss_norm = l2loss(thetas_delta_cu,thetas_cu_zeros)
        loss_beta = l2loss(betas_cu,betas_cu_zeros)
        loss_dct =  l2loss(dct_coef*weight_coeff,coeff_cu_zeros)


        #loss_total = j2d_loss + 0.1*loss_norm + loss_beta + v2d_loss
        loss_total = j2d_loss + 1e-4*v2d_loss + 1e-3*loss_norm + 1e-3*loss_dct + loss_beta
        optimizer.zero_grad()	
        loss_total.backward()
        optimizer.step()
        print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

    return betas_cu.detach().cpu().numpy(),thetas_cu.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()

def main():
    
    smpl_path = '/home/xiul/workspace/TotalDensify/models/smplModel_with_coco25_reg.pkl'
    smplModelGPU = SmplModelTorch(pkl_path=smpl_path,reg_type='coco25')
    smplModelCPU = SmplModel(pkl_path=smpl_path,reg_type='coco25')

    root_path = '/home/xiul/databag/dslr_dance'
    fit_data = dataIO.prepare_data_total_monocular(root_path)
    # print(fit_data)
    #ax.imshow(fit_data['img'])
    
    # plot_vis.plot_coco25_joints(fit_data['joints'],fit_data['joints_weight'],ax,'r')
    # plt.show()
    j2d_tar = fit_data['joints'] 
    j2d_w = fit_data['joints_weight'][:,:,None] 
    j2d_w[:,20:,:] =0
    v2d_tar = fit_data['verts'][:,:,[2,1]]

    v2d_w = fit_data['verts'][:,:,0][:,:,None]
    v2d_w[v2d_w>0] = 1
    v2d_w[v2d_w<0] = 0
    # weight_tar = weight_tar[:,:,None]
    # weight_tar = np.repeat(weight_tar,2,axis=2)
    n_frame = j2d_tar.shape[0]
    print(j2d_tar.shape)
    print(j2d_w.shape)
    print(v2d_tar.shape)
    print(v2d_w.shape)

    
    
    K = np.eye(3)
    R = np.eye(3)
    t = np.zeros((1,3))


    img_size = fit_data['img'][0].shape[1]
    img_height = fit_data['img'][0].shape[0]
    # u = 2 * (u - imsize/2.) / imsize
    # v = 2 * (v - imsize/2.) / imsize

    j2d_tar =2*(j2d_tar - img_size/2.0)/img_size
    v2d_tar =2*(v2d_tar - img_size/2.0)/img_size
    
    cam = [img_size,img_size/2,img_height/2]
    K[0,0] = cam[0]
    K[1,1] = cam[0]
    K[0,2] = cam[1]
    K[1,2] = cam[2]
    K[2,2] = 1
    K = K[None,:,:]
    R = R[None,:,:]
    t = t[None,:,:]




    init_param = {'betas':np.array(fit_data['betas']).astype(np.float32),
                'theta':np.array(fit_data['pose']).reshape(-1,24,3).astype(np.float32),
                'trans':np.array(fit_data['trans'])[:,None,:].astype(np.float32),
                'K':K,
                'R':R,
                't':t,
                'img_size':img_size}


    t0 = time.time()
    betas,thetas,trans = densify2d_adam(smplModelGPU,j2d_tar,j2d_w,v2d_tar,v2d_w,init_param,100,'coco25')
    t1 = time.time()


    print('time passed for 100 frame {}'.format(t1-t0))

    betas_cu_n = torch.tensor(betas).cuda()
    thetas_cu_n = torch.tensor(thetas).cuda()
    trans_cu_n = torch.tensor(trans).cuda()

    v3d_n, j3d_n = smplModelGPU(betas_cu_n, thetas_cu_n, reg_type='coco25')
    #v3d_n = v3d_n/100.0 + trans_cu_n
    #j3d_n = j3d_n/100.0 + trans_cu_n
    #v3d_n = v3d_n/100.0 + trans_cu_n
    #j3d_n = j3d_n/100.0 + trans_cu_n    
    v3d_n = v3d_n/100.0 + trans_cu_n
    j3d_n = j3d_n/100.0 + trans_cu_n

    v2d_proj = projection(v3d_n,torch.tensor(K,dtype=torch.float32).cuda(),torch.tensor(R,dtype=torch.float32).cuda(),torch.tensor(t,dtype=torch.float32).cuda(),img_size)
    j2d_proj = projection(j3d_n,torch.tensor(K,dtype=torch.float32).cuda(),torch.tensor(R,dtype=torch.float32).cuda(),torch.tensor(t,dtype=torch.float32).cuda(),img_size)
    
    v2d_res = v2d_proj.detach().cpu().numpy()
    j2d_res = j2d_proj.detach().cpu().numpy()
    v2d_res = ( v2d_res * img_size)/2.0 + img_size/2
    j2d_res = ( j2d_res * img_size)/2.0 + img_size/2
    
    v2d_tar = ( v2d_tar * img_size)/2.0 + img_size/2
    j2d_tar = ( j2d_tar * img_size)/2.0 + img_size/2
     
    #calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'

    land_marks_pkl = '/home/xiul/workspace/up/models/pose/landmarks.pkl'
    with open(land_marks_pkl) as fio:
        lm = pickle.load(fio)
    
    kps_ = []
    for lk,lim in lm.iteritems():
        kps_.append(lim)


    print(kps_)

    for i in range(n_frame):
        ax.clear()
        ax.set_xlim(0,1920)
        ax.set_ylim(1080,0)

        ax.imshow(fit_data['img'][i])
        ax.scatter(v2d_res[i,kps_,0],v2d_res[i,kps_,1],s=1,c='r')
        ax.scatter(j2d_res[i,:,0],j2d_res[i,:,1],s=3,c='r')
        ax.scatter(v2d_tar[i,:,0],v2d_tar[i,:,1],s=0.1,c='g')
        ax.scatter(j2d_tar[i,:,0],j2d_tar[i,:,1],s=3,c='g')
        plt.savefig('/tmp/ots_{:04d}.png'.format(i),bbox_inches='tight')
        plt.draw()
        plt.pause(0.01)
    


        #raw_input()

if __name__=='__main__':
    main()
