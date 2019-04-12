import sys
import torch
import time
import numpy as np
import numpy.linalg as nlg
import totaldensify.optimizer.std_capture as capture_utils
from totaldensify.model.batch_smpl_torch import SmplModelTorch
import totaldensify.data.dataIO as dataIO
from totaldensify.vis.egl_render import EglRender
from totaldensify.utils.config import cfg
import totaldensify.utils.config as cfg_utils
import cPickle as pickle
import glob
import os.path
import json
import scipy.io
import totaldensify.geometry.geometry_process as geo_utils
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim
if __name__=='__main__':

	cfg_utils.merge_cfg_from_file('../../config/TotalCapture.yaml')
	cfg_utils.assert_and_infer_cfg()

	capUtils = capture_utils.StdCapture(cfg.CAPTURE.MODEL_FEMALE,cfg.CAPTURE.MODEL_MALE,reg_type='total')


	with open('../tests/stage1.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']

	v_all,J_all = capUtils.smplFemaleCpu(betas_n,thetas_n)
	v_all += trans_n

	n_view_batch = 31
	n_frame_batch = 1
	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'
	img_w = 1280.0
	img_h = 720.0
	egl = EglRender(int(img_w),int(img_h))
	

	#load the v_pattern.
	v_pattern = np.loadtxt('../../models/smpl_v_pattern.txt').astype(np.int32)

	# f1----fn , f1-----fn  f1 ------fn 
	#use frame 0
	vis_mask = np.zeros((n_view_batch*n_frame_batch,6890,1),dtype=np.float32)
	vis_img = []
	R_all = np.zeros((n_view_batch*n_frame_batch,3,3))
	t_all = np.zeros((n_view_batch*n_frame_batch,1,3))
	K_all = np.zeros((n_view_batch*n_frame_batch,3,3))
	f = capUtils.smplMaleCpu.f
	with open(calibFile,'r') as fio:
		rawCalibs = json.load(fio)
	cameras = rawCalibs['cameras']
	allPanel = map(lambda x:x['panel'],cameras)
	hdCamIndices = [i for i,x in enumerate(allPanel) if x==0]
	hdCams = [cameras[i] for i in hdCamIndices]

	vc_dummpy = np.zeros((6890,3),dtype=np.float32)

	#step.0 get the visibility map
	for frameId in range(n_frame_batch):
		for camId in range(n_view_batch):
			vt = v_all[frameId]
			R = np.array(hdCams[camId]['R'])
			t = np.array(hdCams[camId]['t']).T
			K = np.array(hdCams[camId]['K'])*1280.0/1920.0
			R_all[camId*n_frame_batch+frameId] = R
			t_all[camId*n_frame_batch+frameId] = t
			K_all[camId*n_frame_batch+frameId] = K
			vn = geo_utils.vertices_normals(f,vt)
			vis_image,vis_depth = egl.render_obj(vt,vc_dummpy,vn,f,R,t.T,K)
			vis_img.append(vis_depth)
	vis_img = np.array(vis_img)
	vis_img_cu = torch.from_numpy(vis_img[:,:,:,None].transpose(0,3,1,2).astype(np.float32)).cuda()

	vt_cu = torch.tensor(v_all[:n_frame_batch].repeat(n_view_batch,axis=0),dtype=torch.float32).cuda()
	K_cu = torch.tensor(K_all,dtype=torch.float32).cuda()
	R_cu = torch.tensor(R_all,dtype=torch.float32).cuda()
	t_cu = torch.tensor(t_all,dtype=torch.float32).cuda()

	vt_2d_cu = geo_utils.projection_cuda(vt_cu,K_cu,R_cu,t_cu,img_w,img_h)
	vt_2d_cpu = vt_2d_cu.cpu().numpy()
	vt_2d_grid = vt_2d_cu[:,:,:2]
	vt_2d_grid = vt_2d_grid[:,v_pattern,:]
	#the sample grid

	depth_2d = F.grid_sample(vis_img_cu,vt_2d_grid,mode='nearest')
	depth_2d = depth_2d.cpu().numpy().transpose(0,2,3,1)[:,:,:,0]

	depth_2d_vts = vt_2d_cpu[:,v_pattern,2]
	vis_diff = nlg.norm(depth_2d_vts - depth_2d,ord=np.inf,axis=2)
	vis_mask = np.zeros((n_view_batch*n_frame_batch,6890),dtype=np.float32)
	vis_mask[vis_diff<5e-1] = 1

	#visibility for each viewpoints

	img_target = []
	for frameId in range(n_frame_batch):
		for view_id in range(n_view_batch):
			img_path = '/home/xiul/databag/dome_sptm/171204_pose6/images/00_{:02d}_{:08d}.jpg'.format(view_id,frameId+500)
			if os.path.isfile(img_path):
				img0 = cv2.imread(img_path).astype(np.float32)
			else:
				img0 = np.zeros((720,1280,3),dtype=np.float32)
				vis_mask[view_id*n_frame_batch+frameId,:] = 0
			img_target.append(img0)
	img_target = np.array(img_target)
	img_target_cu = torch.tensor(img_target.transpose(0,3,1,2)/255.0,dtype=torch.float32).cuda()
	vis_mask_num = np.sum(vis_mask,axis=0)
	#vertices with no more than 3 viewpoint will be set as invisibile
	vis_mask[:,vis_mask_num<3] = 0




	# img_2d = F.grid_sample(img_target_cu,vt_2d_grid,mode='nearest')
	# img_2d = img_2d.cpu().numpy().transpose(0,2,3,1)
	init_thetas_cu = torch.tensor(thetas_n[:n_frame_batch],dtype=torch.float32).cuda()
	delta_thetas_cu = torch.zeros_like(init_thetas_cu,dtype=torch.float32,requires_grad=True,device='cuda').cuda()
	zero_thetas_cu = torch.zeros_like(init_thetas_cu,dtype=torch.float32).cuda()

	init_betas_cu = torch.tensor(betas_n[0][None,:],dtype=torch.float32).cuda()
	delta_betas_cu = torch.zeros_like(init_betas_cu,dtype=torch.float32,requires_grad=True,device='cuda').cuda()
	zero_betas_cu = torch.zeros_like(init_betas_cu,dtype=torch.float32).cuda()

	trans_cu = torch.tensor(trans_n[:n_frame_batch],dtype=torch.float32,requires_grad=True,device='cuda').cuda()


	thetas_cu = init_thetas_cu + delta_thetas_cu
	thetas_cu_batch = thetas_cu.repeat(n_view_batch,1,1)

	betas_cu = init_betas_cu + delta_betas_cu
	betas_cu_batch = betas_cu.repeat(n_view_batch*n_frame_batch,1)

	trans_cu_batch = trans_cu.repeat(n_view_batch,1,1)

	cano_tex = torch.zeros(1,3,6890,requires_grad=True,device='cuda',dtype=torch.float32).cuda()
	
	

	l2loss = torch.nn.MSELoss().cuda()


	#optimizer = torch.optim.Adam([cano_tex],lr=0.01)
	optimizer = torch.optim.Adam([{'params':delta_betas_cu,'lr':1e-10},
                                  {'params':delta_thetas_cu,'lr':1e-10},
                                  {'params':trans_cu,'lr':1e-10},
								  {'params':cano_tex,'lr':1}],lr=1e-1)
	# delta_betas_cu.requires_grad=False
	# delta_thetas_cu.requires_grad=False
	# trans_cu.requires_grad=False

	#vis_mask_cu = torch.tensor(vis_mask[:,None,:,:].repeat(3,axis=1),dtype=torch.float32).cuda()
	
	vis_mask_cu = torch.tensor(vis_mask[:,None,:,None].repeat(3,axis=1),dtype=torch.float32).cuda()

	for i_iter in range(400):
		cano_tex_batch = cano_tex.repeat(n_view_batch*n_frame_batch,1,1)
		thetas_cu = init_thetas_cu + delta_thetas_cu
		thetas_cu_batch = thetas_cu.repeat(n_view_batch,1,1)

		betas_cu = init_betas_cu + delta_betas_cu
		betas_cu_batch = betas_cu.repeat(n_view_batch*n_frame_batch,1)

		trans_cu_batch = trans_cu.repeat(n_view_batch,1,1)

		v3d_batch, j3d_batch = capUtils.smplFemaleCuda(betas_cu_batch,thetas_cu_batch)
		v3d_batch = v3d_batch + trans_cu_batch
		v2d_cu_batch = geo_utils.projection_cuda(v3d_batch,K_cu,R_cu,t_cu,img_w,img_h)

		#vt_2d_cu = geo_utils.projection(vt_cu,K_cu,R_cu,t_cu,img_w,img_h)
		vt_2d_grid_batch = v2d_cu_batch[:,v_pattern,:2]
		rep_color_cu = F.grid_sample(img_target_cu,vt_2d_grid)
		cano_tex_batch_patch = cano_tex_batch[:,:,v_pattern]


		loss_tex = l2loss(rep_color_cu*vis_mask_cu,cano_tex_batch_patch*vis_mask_cu)

		loss_total = loss_tex 
		#+ 1e1*loss_shape + 1e1*loss_reg
		optimizer.zero_grad()
		loss_total.backward()
		optimizer.step()
		print('loss is {}'.format(loss_total.item()))
		
	
	print(cano_tex.shape)
	cano_tex_cpu = cano_tex.detach().cpu().numpy()[0,:,:].transpose(1,0)
	print(cano_tex_cpu.shape)
	np.savetxt('best_texture.txt',cano_tex_cpu)