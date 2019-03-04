import sys
import torch
import torch.nn
import numpy as np
import time
#import totaldensify.optimizer.torch_smplify as smplify_util
from totaldensify.model.batch_smpl_torch import SmplModelTorch
from totaldensify.model.batch_smpl import SmplModel
#from totaldensify.vis.glut_viewer import glut_viewer
import cPickle as pickle
import totaldensify.data.dataIO as dataIO
import matplotlib.pyplot as plt
import totaldensify.vis.plot_vis as plot_vis
import neural_renderer as nr
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

def densify2d_adam(totalModel,j2d_t,inds_t,face,tex,render,init_param,n_iter,reg_type):

    n_batch = j2d_t.shape[0]

    betas_cu = torch.tensor(
        init_param['betas'],requires_grad=True, device='cuda').cuda()
    thetas_cu = torch.tensor(
        init_param['theta'],requires_grad=True, device='cuda').cuda()
    trans_cu = torch.tensor(
        init_param['trans'],requires_grad=True, device = 'cuda').cuda()
    cam_K_cu = torch.tensor(init_param['K'],dtype=torch.float32).cuda()
    cam_R_cu = torch.tensor(init_param['R'],dtype=torch.float32).cuda()
    cam_t_cu = torch.tensor(init_param['t'],dtype=torch.float32).cuda()
    weights = torch.tensor(init_param['weight'],dtype=torch.float32).cuda()

    img_size = init_param['img_size']


    thetas_cu_zeros = torch.zeros_like(thetas_cu).cuda()
    betas_cu_zeros = torch.zeros_like(betas_cu).cuda()

    #weight_cu = self.weight_coco25_cu.repeat(n_batch, 1, 1)

    j2d_t = torch.tensor(j2d_t,dtype=torch.float32).cuda() 


    optimizer = torch.optim.Adam([{'params':betas_cu},
                                    {'params':thetas_cu},
                                    {'params':trans_cu}], lr=0.1)
    l2loss = torch.nn.MSELoss().cuda()
    #bceloss = torch.nn.BCELoss().cuda()
    


    for i in range(n_iter):
        v3d_pred, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
        j3d_pred = j3d_pred /100.0 + trans_cu
        v3d_pred = v3d_pred /100.0 + trans_cu
        #j3d_pred = j3d_pred / 100.0
        v3d_to_render = v3d_pred.view(1,-1,3)
        #print(v3d_to_render.dtype)

        #images = rendered(vts_float,model_faces_cu,tex,None,K_cu,R_cu,t_cu,None,img_size)

        
        j2d_pred = projection(j3d_pred,cam_K_cu,cam_R_cu,cam_t_cu,img_size)[:,:,:2]
        j2d_loss = l2loss(j2d_pred*weights, j2d_t*weights)
        
    
        loss_norm = l2loss(thetas_cu,thetas_cu_zeros)
    
        loss_beta = l2loss(betas_cu,betas_cu_zeros)
        if i==80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1
        if i<=80:
            loss_total = j2d_loss + 0.001*loss_norm + 1*loss_beta
        else:
            rendered_img = render(v3d_to_render,face,tex,None,cam_K_cu,cam_R_cu,cam_t_cu,None,img_size)
            #dense_error = (rendered_img - inds_t)**2
            #log_prob = torch.exp(dense_error)
            dense_loss = l2loss(rendered_img,inds_t)
            #dense_loss = torch.sum(log_prob)
            loss_total = j2d_loss + 0.001*loss_norm + loss_beta + dense_loss*1e-4
        optimizer.zero_grad()	
        loss_total.backward()
        optimizer.step()
        print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

    return betas_cu.detach().cpu().numpy(),thetas_cu.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()

def mannual_correct(inds_img):
    inds_dict_img = [10,7,4,9,8,6,3,11,5,2]
    inds_dict_target = [3,2,8,5,6,1,10,4,9,7]

    inds_img_c0 = inds_img[:, :, 0]
    mask_img = np.zeros(inds_img_c0.shape, inds_img_c0.dtype)
    for img_i,target_i in zip(inds_dict_img,inds_dict_target):
        mask_img[inds_img_c0==img_i] = target_i
    #re-arrange idx as 1,2,3 counting numbers
    mask_img = np.dstack((mask_img, mask_img, mask_img))

    return mask_img


def test_load_data_multi(img_path,root_path):
    
    smpl_path = '/home/xiul/workspace/TotalDensify/models/smplModel_with_coco25_reg.pkl'
    smplModelGPU = SmplModelTorch(pkl_path=smpl_path,reg_type='coco25')
    smplModelCPU = SmplModel(pkl_path=smpl_path,reg_type='coco25')

    fit_data = dataIO.prepare_data_total(root_path,img_path)
    # print(fit_data)
    #ax.imshow(fit_data['img'])
    
    # plot_vis.plot_coco25_joints(fit_data['joints'],fit_data['joints_weight'],ax,'r')
    # plt.show()
    j2d_tar = fit_data['joints']
    weight_tar = fit_data['joints_weight']
    for w in weight_tar:
        w[19:] = 0

    weight_tar = weight_tar[:,:,None]
    weight_tar = np.repeat(weight_tar,2,axis=2)




    K = np.eye(3)
    R = np.eye(3)
    t = np.zeros((1,3))
    #cam = fit_data['cam'][0]

    #print(fit_data['joints'].shape
    num_body = len(fit_data['cam'])

    img_size = fit_data['img'].shape[1]
    img_height = fit_data['img'].shape[0]

    cam = [img_size,img_size/2,img_height/2]
    K[0,0] = cam[0]
    K[1,1] = cam[0]
    K[0,2] = cam[1]
    K[1,2] = cam[2]
    K[2,2] = 1
    K = K[None,:,:]
    R = R[None,:,:]
    t = t[None,:,:]


    num_vertices = smplModelCPU.v_template.shape[0]
    model_faces = smplModelCPU.f
    for i in range(1,num_body):
        model_faces = np.vstack((model_faces,smplModelCPU.f+i*num_vertices))
    print(model_faces.shape)
    

    f_num = smplModelCPU.f.shape[0]

    model_faces_cu = torch.tensor(model_faces[None,:,:].astype(np.int32),dtype=torch.int32).cuda()

    tex_np = np.zeros((1,model_faces_cu.shape[1],2,2,2,3))
    for i in range(num_body):
        tex_np[0,i*f_num:(i+1)*f_num,:] = c_map(i/11.0)[:3]
    tex = torch.tensor(tex_np,dtype=torch.float32).cuda()
    inds_box = np.zeros((img_size,img_size,3),dtype=np.float32)
    inds_img = mannual_correct(fit_data['inds_img'])
    inds_cmap = np.zeros(inds_img.shape,np.float32)

    for idx in range(num_body):
        inds_cmap[inds_img[:,:,0]==(idx+1),:] = c_map(idx/11.0)[:3] 

    #inds_img = np.flip(inds_img,0)
    inds_box[:inds_img.shape[0],:,:] = inds_cmap
    inds_box = np.flip(inds_box,0)

    import cv2
    inds_box = cv2.resize(inds_box,(256,256))

    inds_box = inds_box.transpose(2,0,1)
    # ax.imshow(inds_box[0,:,:])
    # plt.show()

    inds_t = torch.tensor(inds_box[None,:,:,:],dtype=torch.float32).cuda()


    rendered = nr.Renderer(image_size=256).cuda()
    rendered.light_intensity_directional = 0.0
    rendered.light_intensity_ambient = 1.0



    init_param = {'betas':np.array(fit_data['betas_init']).astype(np.float32),
                'theta':np.array(fit_data['pose_init']).reshape(-1,24,3).astype(np.float32),
                'trans':np.array(fit_data['trans_init'])[:,None,:].astype(np.float32),
                'K':K,
                'R':R,
                't':t,
                'img_size':img_size,
                'weight':weight_tar}

    #j2d_tar[:,:,0] = 2 * (j2d_tar[:,:,0] - img_size/2.) / img_size
    j2d_tar = 2 * (j2d_tar - img_size/2.) / img_size
    #print(np.array(fit_data['trans_init']).shape)
    fit = True


    if not fit:
        betas_init_cu = torch.tensor(init_param['betas']).cuda()
        pose_init_cu = torch.tensor(init_param['theta']).cuda()
        trans_init_cu = torch.tensor(init_param['trans']).cuda()
    else:
        betas,thetas,trans = densify2d_adam(smplModelGPU,j2d_tar,inds_t,model_faces_cu,tex,rendered,init_param,120,'coco25')
        betas_init_cu = torch.tensor(betas).cuda()
        pose_init_cu = torch.tensor(thetas).cuda()
        trans_init_cu = torch.tensor(trans).cuda()


    vts,jts = smplModelGPU(betas_init_cu,pose_init_cu,reg_type='coco25')
    vts = vts/100.0
    jts = jts/100.0
    vts += trans_init_cu
    jts += trans_init_cu

    #def projection(vt,K,R,t,imsize):
    proj_2d = projection(jts,torch.tensor(K,dtype=torch.float32).cuda(),torch.tensor(R,dtype=torch.float32).cuda(),torch.tensor(t,dtype=torch.float32).cuda(),img_size)
    proj_2d_cpu = proj_2d.detach().cpu().numpy()*img_size/2 + img_size/2

    ax.imshow(fit_data['img'])

    # 	#j2d_tar

    
    #model_faces = smplModelCPU.f[None,:,:]

    K_cu = torch.tensor(K,dtype=torch.float32).cuda()
    R_cu = torch.tensor(R,dtype=torch.float32).cuda()
    t_cu = torch.tensor(t,dtype=torch.float32).cuda()
    vts_float = vts.view(-1,3)
    vts_float = vts_float.type(torch.float32)[None,:,:]

    images = rendered(vts_float,model_faces_cu,tex,None,K_cu,R_cu,t_cu,None,img_size)
    #image_cpu = images.detach().cpu().numpy()[0]

    image_cpu = images.detach().cpu().numpy()[0].transpose((1,2,0))
    image_cpu = np.flip(image_cpu,0)
    image_cpu = cv2.resize(image_cpu,(img_size,img_size))

    image_cpu = image_cpu[:img_height,:,:]
    alpha_cpu = np.ones((image_cpu.shape[0],image_cpu.shape[1]))
    sum_color = np.sum(image_cpu,axis=2)

    alpha_cpu[sum_color<=1e-3] = 0
    rgba_image = np.dstack((image_cpu,alpha_cpu[:,:,None]))
    ax.imshow(rgba_image,alpha=0.5)
    #ax[1].imshow(image_cpu)
    plot_w = fit_data['joints_weight'][0]
    plot_w[19:] = 0
    #ax.scatter(man_u,man_v,s=5)
    for idx,proj in enumerate(proj_2d_cpu):
        plot_w = fit_data['joints_weight'][0]
        plot_w[19:] = 0
        plot_vis.plot_coco25_joints(proj[:,:2],plot_w,ax,c_map(idx/11.0))

    # for j2d,plot_w in zip(j2d_tar,fit_data['joints_weight']):
    # 	plot_vis.plot_coco25_joints(j2d*img_size/2+img_size/2,plot_w,ax,'g')
    # print(images.shape)
    # image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    # print(image.shape)
    # img_uint8 = (255*image).astype(np.uint8)
    # ax.imshow(img_uint8)
    ax.set_xticks(())
    ax.set_yticks(())
    # fig,ax2 = plt.subplots(3,2)
    # for ch in range(3):
    #     ax2[ch][0].imshow(image_cpu[ch,:,:])
    #     ax2[ch][1].imshow(inds_box[ch,:,:])
    
    
    plt.show()
    
    
    
    #writer.append_data((255*image).astype(np.uint8))

if __name__=='__main__':
    img_path = '/home/xiul/databag/denseFusion/images/run.jpg'
    root_path = '/home/xiul/databag/denseFusion'
    test_load_data_multi(img_path,root_path)