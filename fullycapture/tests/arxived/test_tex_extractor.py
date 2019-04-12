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
import glob
import os.path
import torch.nn as nn
from skimage.io import imread, imsave
import tqdm
import imageio
import cv2

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

class Model(nn.Module):
    def __init__(self, verts, faces,ref_images,K,R,t):
        super(Model, self).__init__()
        #vertices, faces = nr.load_obj(filename_obj)
        self.register_buffer('vertices', verts)
        self.register_buffer('faces', faces)
        # create textures
        #K = K*1280/1920.0
        self.register_buffer('K',K)
        self.register_buffer('R',R)
        self.register_buffer('t',t)

        n_batch = verts.shape[0]
        self.n_batch = n_batch
        
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        
        self.textures = nn.Parameter(textures)
        self.image_size = ref_images[0].shape[1]
        image_short = ref_images[0].shape[0]
        #image_src = np.zeros((n_batch,self.image_size,self.image_size,3),dtype=np.float32)
        image_tar = np.zeros((n_batch,256,256,3),dtype=np.float32)

        for i in range(n_batch):
            image_src = np.zeros((self.image_size,self.image_size,3))
            image_src[:image_short,:,:] = ref_images[i].astype(np.float32)/255.0
            image_src = np.flip(image_src,axis=0)
            image_tar[i] = cv2.resize(image_src,(256,256))
        image_ref = torch.tensor(image_tar.transpose(0,3,1,2),dtype=torch.float32)

        self.register_buffer('image_ref', image_ref)
        # setup renderer
        renderer = nr.Renderer()
        #renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer
        print('model build')
    def forward(self):
        #self.renderer.eye = nr.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        texture_to_render = self.textures.repeat(self.n_batch,1,1,1,1,1)
        image = self.renderer(self.vertices, self.faces,torch.tanh(texture_to_render),None,self.K,self.R,self.t,None,self.image_size)
        loss = torch.sum((image - self.image_ref) ** 2)
        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


#import neural_render as nr
fig, ax = plt.subplots()

def main():
    smpl_path = '/home/xiul/workspace/TotalDensify/models/smplModel_with_coco25_reg.pkl'
    smplModel = SmplModelTorch(pkl_path=smpl_path,reg_type='coco25')
    smplModelCPU = SmplModel(pkl_path=smpl_path,reg_type='coco25')
    
    calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'
    root_image = '/home/xiul/databag/dome_sptm/171204_pose6/images'
    

    model_path = '/home/xiul/databag/dome_sptm/171204_pose6/gt_pkl'
    model_pkls = glob.glob(os.path.join(model_path, '*.pkl'))
    model_pkls.sort()

    frame_id = 50
    
    view_nodes = range(21)+range(22,31)
    print(len(view_nodes))
    K,R,t = dataIO.load_dome_calibs(calibFile,view_nodes)
    ref_images = dataIO.load_dome_images_wrt_frame(root_image,500+frame_id,view_nodes)
    # cv2.imshow('ref_image',ref_images[0])
    # cv2.waitKey(-1)
    n_view = len(view_nodes)
    #frame_id = 80
    #model_pkls = [model_pkls[65]]
    #model_pkls = model_pkls[:100]
    n_batch = len(model_pkls)
    
    with open('smpl.pkl') as fio:
        initDD = pickle.load(fio)
    #betas_zero = np.zeros((1, 10)).astype(np.float32)
    betas_zero = initDD['betas']
    thetas_zero = initDD['theta']
    trans_zero = initDD['trans']
    #print(betas_n)

    betas_cu_n = torch.tensor(betas_zero,dtype=torch.float32).cuda()
    thetas_cu_n = torch.tensor(thetas_zero,dtype=torch.float32).cuda()
    trans_cu_n = torch.tensor(trans_zero,dtype=torch.float32).cuda()

    v3d_n, j3d_n = smplModel(betas_cu_n, thetas_cu_n, reg_type='coco25')
    v3d_n = v3d_n + trans_cu_n
    #vertices
    v3d_n0 = v3d_n[frame_id][None,:,:].repeat(n_view,1,1)/100.0




    model_faces = smplModelCPU.f

    model_faces_cu = torch.tensor(model_faces[None,:,:].repeat(n_view,axis=0).astype(np.int32),dtype=torch.int32).cuda()



    K_cu = torch.tensor(K*1280.0/1920.0,dtype=torch.float32).cuda()
    R_cu = torch.tensor(R,dtype=torch.float32).cuda()
    t_cu = torch.tensor(t,dtype=torch.float32).cuda()/100.0
    
    v2d = projection(v3d_n0,K_cu,R_cu,t_cu,1280)
    # for i in range(n_view):

    #     v2d_0 = v2d.detach().cpu().numpy()[i]
    #     ax.clear()
    #     ax.imshow(ref_images[i][:,:,::-1])
    #     v2d_0 = v2d_0 * 1280 / 2.0 + 1280/2
    #     ax.scatter(v2d_0[:,0],v2d_0[:,1],s=1)
    #     plt.pause(0.01)
    #     plt.draw()
    #     raw_input()
    #     #plt.show()
    model = Model(v3d_n0,model_faces_cu,ref_images,K_cu,R_cu,t_cu)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5,0.999))
    
    loop = tqdm.tqdm(range(50))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()


    tex = model.textures.detach().cpu().numpy()
    with open('tex.pkl','wb') as fio:
        pickle.dump(tex,fio)
        
    # textures = torch.ones(n_view, model_faces_cu.shape[1], 2, 2, 2, 3, dtype=torch.float32).cuda()*0.5


    # rendered = nr.Renderer(image_size=256).cuda()
    # rendered.light_intensity_directional = 0.0
    # rendered.light_intensity_ambient = 1.0      

    # images = rendered(v3d_n0, model_faces_cu,textures,None,K_cu,R_cu,t_cu/100.0,None,1280)
    # images = images.detach().cpu().numpy()
    # print(images.shape)
    # image = images[0].transpose((1, 2, 0))
    
    # plt.imshow(image)
    # plt.show()
    # # draw object
    # loop = tqdm.tqdm(range(n_view))
    # for num, view_id in enumerate(loop):
    #     loop.set_description('Drawing')
    #     #model.renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
    #     tex_to_render = model.textures.repeat(n_view,1,1,1,1,1)
    #     #print(model.image_size)
    #     images = rendered(v3d_n0,model_faces_cu,torch.tanh(tex_to_render),None,K_cu,R_cu,t_cu,None,1280)
    #     image = images.detach().cpu().numpy()[view_id].transpose((1, 2, 0))
    #     imsave('/tmp/_tmp_%04d.png' % num, image)

def render_main():
    smpl_path = '/home/xiul/workspace/TotalDensify/models/smplModel_with_coco25_reg.pkl'
    smplModel = SmplModelTorch(pkl_path=smpl_path,reg_type='coco25')
    smplModelCPU = SmplModel(pkl_path=smpl_path,reg_type='coco25')
    
    calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'
    root_image = '/home/xiul/databag/dome_sptm/171204_pose6/images'
    

    model_path = '/home/xiul/databag/dome_sptm/171204_pose6/gt_pkl'
    model_pkls = glob.glob(os.path.join(model_path, '*.pkl'))
    model_pkls.sort()

    view_nodes = range(21)+range(22,31)
    print(len(view_nodes))
    K,R,t = dataIO.load_dome_calibs(calibFile,view_nodes)

    # cv2.imshow('ref_image',ref_images[0])
    # cv2.waitKey(-1)
    n_view = len(view_nodes)
    #frame_id = 80
    #model_pkls = [model_pkls[65]]
    #model_pkls = model_pkls[:100]
    n_batch = len(model_pkls)
    
    with open('smpl_sptm.pkl') as fio:
        initDD = pickle.load(fio)
    #betas_zero = np.zeros((1, 10)).astype(np.float32)
    betas_zero = initDD['betas']
    thetas_zero = initDD['theta']
    trans_zero = initDD['trans']
    #print(betas_n)

    betas_cu_n = torch.tensor(betas_zero,dtype=torch.float32).cuda()
    thetas_cu_n = torch.tensor(thetas_zero,dtype=torch.float32).cuda()
    trans_cu_n = torch.tensor(trans_zero,dtype=torch.float32).cuda()

    v3d_n, j3d_n = smplModel(betas_cu_n, thetas_cu_n, reg_type='coco25')
    v3d_n = v3d_n + trans_cu_n
    v3d_n_render = v3d_n/100.0

    #vertices
    # v3d_n0 = v3d_n[frame_id][None,:,:].repeat(n_view,1,1)/100.0
    model_faces = smplModelCPU.f

    model_faces_cu = torch.tensor(model_faces[None,:,:].repeat(n_batch,axis=0).astype(np.int32),dtype=torch.int32).cuda()


    K_cu = torch.tensor(K[0][None,:,:]*1280.0/1920.0,dtype=torch.float32).cuda()
    R_cu = torch.tensor(R[0][None,:,:],dtype=torch.float32).cuda()
    t_cu = torch.tensor(t[0][None,:,:],dtype=torch.float32).cuda()/100.0
    

    with open('tex.pkl') as fio:
        tex_data = pickle.load(fio)
       
    textures = torch.tensor(tex_data,dtype=torch.float32).cuda()
    textures_render = textures.repeat(n_batch,1,1,1,1,1)

    rendered = nr.Renderer(image_size=256).cuda()
    rendered.light_intensity_directional = 0.0
    rendered.light_intensity_ambient = 1.0      

    images = rendered(v3d_n_render, model_faces_cu,torch.tanh(textures_render),None,K_cu,R_cu,t_cu,None,1280)
    images = images.detach().cpu().numpy()
    print(images.shape)
    

    for i in range(n_batch):
        image = images[i].transpose((1, 2, 0))[:,:,::-1]
        image = np.flip(image,axis=0)

        imsave('/tmp/_tmp_{:04d}.png'.format(i), image)
    #    plt.imshow(image[:,:,::-1])
    #     plt.draw()
    #     plt.pause(0.01)
    #     raw_input()
        # draw object
    # loop = tqdm.tqdm(range(n_view))
    # for num, view_id in enumerate(loop):
    #     loop.set_description('Drawing')
    #     #model.renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
    #     tex_to_render = model.textures.repeat(n_view,1,1,1,1,1)
    #     #print(model.image_size)
    #     images = rendered(v3d_n0,model_faces_cu,torch.tanh(tex_to_render),None,K_cu,R_cu,t_cu,None,1280)
    #     image = images.detach().cpu().numpy()[view_id].transpose((1, 2, 0))
    #     imsave('/tmp/_tmp_%04d.png' % num, image)

    make_gif('test_sptm.gif')

if __name__=='__main__':
    #main()
    render_main()