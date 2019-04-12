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

	model_path = cfg.CAPTURE.MODEL_FEMALE
	capUtils = capture_utils.StdCapture(cfg.CAPTURE.MODEL_FEMALE,cfg.CAPTURE.MODEL_MALE,reg_type='total')

	
	with open('../tests/stage1_171204_pose3.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']

	v_all,J_all = capUtils.smplFemaleCpu(betas_n,thetas_n)
	v_all += trans_n


	n_view_batch = 31
	n_frame_batch = 3

	# J_all += trans_n

	init_thetas_cu = torch.tensor(thetas_n[:n_frame_batch],dtype=torch.float32).cuda()
	delta_thetas_cu = torch.zeros_like(init_thetas_cu,dtype=torch.float32,requires_grad=True,device='cuda').cuda()
	zero_thetas_cu = torch.zeros_like(init_thetas_cu,dtype=torch.float32).cuda()

	betas_init_cu = torch.tensor(betas_n[0][None,:],dtype=torch.float32).cuda()
	betas_delta_cu = torch.zeros_like(betas_init_cu,dtype=torch.float32,requires_grad=True,device='cuda').cuda()
	betas_zero_cu = torch.zeros_like(betas_init_cu,dtype=torch.float32).cuda()

	trans_cu = torch.tensor(trans_n[:n_frame_batch],dtype=torch.float32).cuda()


	thetas_cu = init_thetas_cu + delta_thetas_cu
	thetas_cu_batch = thetas_cu.repeat(n_view_batch,1,1)

	betas_cu = betas_init_cu + betas_delta_cu
	betas_cu_batch = betas_cu.repeat(n_view_batch*n_frame_batch,1)

	trans_cu_batch = trans_cu.repeat(n_view_batch,1,1)

	# f1----fn , f1-----fn  f1 ------fn 

	#use frame 0

	calibFile = '/home/xiul/databag/dome_sptm/171204_pose3/calibration_171204_pose3.json'
	img_w = 1280.0
	img_h = 720.0
	egl = EglRender(int(img_w),int(img_h))


	with open(calibFile,'r') as fio:
		rawCalibs = json.load(fio)
	cameras = rawCalibs['cameras']
	allPanel = map(lambda x:x['panel'],cameras)
	hdCamIndices = [i for i,x in enumerate(allPanel) if x==0]
	hdCams = [cameras[i] for i in hdCamIndices]


	#step.0 get the visibility map
	vis_mask = np.zeros((n_view_batch*n_frame_batch,6890,1),dtype=np.float32)
	vis_img = []

	#vt = v_all[0]
	#vc = np.ones((6890,3),dtype=np.float32)

	R_all = np.zeros((n_view_batch*n_frame_batch,3,3))
	t_all = np.zeros((n_view_batch*n_frame_batch,1,3))
	K_all = np.zeros((n_view_batch*n_frame_batch,3,3))
	vc_dummpy = np.zeros((6890,3),dtype=np.float32)

	f = capUtils.smplFemaleCpu.f

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


	vt_2d_cu = geo_utils.projection(vt_cu,K_cu,R_cu,t_cu,img_w,img_h)
	vt_2d_grid = vt_2d_cu[:,:,None,:2]


	depth_2d = F.grid_sample(vis_img_cu,vt_2d_grid,mode='nearest')
	depth_2d = depth_2d.cpu().numpy().transpose(0,2,1,3)[:,:,:,0]

	vt_2d_cpu = vt_2d_cu.cpu().numpy()

	vis_diff = nlg.norm(depth_2d - vt_2d_cpu[:,:,2][:,:,None],ord=np.inf,axis=2)[:,:,None]
	vis_mask[vis_diff<0.1] = 1
	#the visibiliy mask

	#try the render here.
	# fig,ax = plt.subplots(1)
	
#load init
	#cano_np = np.loadtxt('./stage2_texture.txt')
	#cano_np = cano_np[None,:,:,None].transpose(0,2,1,3)

	cano_tex = torch.zeros(1,3,6890,1,requires_grad=True,device='cuda',dtype=torch.float32).cuda()
	#cano_tex = torch.tensor(cano_np,requires_grad=True,device='cuda',dtype=torch.float32).cuda()
	vis_mask_cu = torch.tensor(vis_mask[:,None,:,:].repeat(3,axis=1),dtype=torch.float32).cuda()
	
	#read images
	img_target = []

	for frameId in range(n_frame_batch):
		for view_id in range(n_view_batch):
			img_path = '/home/xiul/databag/dome_sptm/171204_pose3/images/00_{:02d}_{:08d}.jpg'.format(view_id,frameId+400)
			if os.path.isfile(img_path):
				img0 = cv2.imread(img_path).astype(np.float32)
			else:
				img0 = np.zeros((720,1280,3),dtype=np.float32)
				vis_mask_cu[view_id*n_frame_batch+frameId] = 0

			img_target.append(img0)
		print('load frame {}'.format(frameId))
	img_target = np.array(img_target)

	img_target_cu = torch.tensor(img_target.transpose(0,3,1,2)/255.0,dtype=torch.float32).cuda()

	l2loss = torch.nn.MSELoss().cuda()


	#optimizer = torch.optim.Adam([cano_tex],lr=0.01)
	optimizer = torch.optim.Adam([{'params':betas_delta_cu,'lr':1e-6},
                                  {'params':delta_thetas_cu,'lr':1e-6},
                                  {'params':trans_cu,'lr':1e-6},
								  {'params':cano_tex,'lr':1e-1}],lr=1e-1)
	
	#optimizer = torch.optim.Adam([{'params':cano_tex,'lr':1e-1}],lr=1e-1)

	betas_delta_cu.requires_grad = False
	delta_thetas_cu.requires_grad = False
	trans_cu.requires_grad = False

	#need to sparse sample 
	# vis_mask_cu[:,::2,:] = 0
	# vis_mask_cu[:,::3,:] = 0
	# vis_mask_cu[:,::5,:] = 0
	

	for i_step in range(8):
		if i_step % 2 == 1:
			betas_delta_cu.requires_grad = True
			delta_thetas_cu.requires_grad = True
			trans_cu.requires_grad = True
			cano_tex.requires_grad = True
		else:
			betas_delta_cu.requires_grad = False
			delta_thetas_cu.requires_grad = False
			trans_cu.requires_grad = False
			cano_tex.requires_grad = True
		for i_iter in range(25):
			cano_tex_batch = cano_tex.repeat(n_view_batch*n_frame_batch,1,1,1)
			thetas_cu = init_thetas_cu + delta_thetas_cu
			thetas_cu_batch = thetas_cu.repeat(n_view_batch,1,1)

			betas_cu = betas_init_cu + betas_delta_cu
			betas_cu_batch = betas_cu.repeat(n_view_batch*n_frame_batch,1)

			trans_cu_batch = trans_cu.repeat(n_view_batch,1,1)

			v3d_batch, j3d_batch = capUtils.smplMaleCuda(betas_cu_batch,thetas_cu_batch)
			v3d_batch = v3d_batch + trans_cu_batch
			v2d_cu_batch = geo_utils.projection(v3d_batch,K_cu,R_cu,t_cu,img_w,img_h)
			#vt_2d_cu = geo_utils.projection(vt_cu,K_cu,R_cu,t_cu,img_w,img_h)
			vt_2d_grid = v2d_cu_batch[:,:,None,:2]
			rep_color_cu = F.grid_sample(img_target_cu,vt_2d_grid)

			loss_tex = l2loss(rep_color_cu*vis_mask_cu,cano_tex_batch*vis_mask_cu)
			#loss_shape = l2loss(betas_delta_cu,betas_zero_cu)
			#loss_reg = l2loss(delta_thetas_cu,zero_thetas_cu)

			loss_total = loss_tex 
			#+ 1e1*loss_shape + 1e1*loss_reg
			optimizer.zero_grad()
			loss_total.backward()
			optimizer.step()
			print('loss is {}'.format(loss_total.item()))
		

	cano_tex_cpu = cano_tex.detach().cpu().numpy()[0,:,:,0].transpose(1,0)
	# for camId in range(n_view_batch):
	# 	R = R_all[camId]
	# 	t = t_all[camId]
	# 	K = K_all[camId]

	# 	vn = geo_utils.vertices_normals(f,vt)
	# 	vis_image,vis_depth = egl.render_obj(vt,cano_tex_cpu,vn,f,R,t.T,K)
	# 	cv2.imshow('tex_image',vis_image)
	# 	cv2.imshow('color_image',img_target[camId].astype(np.uint8))
	# 	cv2.waitKey(-1)
	thetas_cu_batch_cpu = thetas_cu_batch.detach().cpu().numpy()[:n_frame_batch]
	betas_cu_batch_cpu = betas_cu_batch.detach().cpu().numpy()[:n_frame_batch]
	trans_cu_batch_cpu = trans_cu_batch.detach().cpu().numpy()[:n_frame_batch]
	
	print(thetas_cu_batch_cpu.shape)

	# betas_n = dd['betas']
	# thetas_n = dd['pose']
	# trans_n = dd['trans']
	
	with open('stage2_171204_pose3.pkl','wb') as fio:
		pickle.dump({'pose':thetas_cu_batch_cpu,'betas':betas_cu_batch_cpu,'trans':trans_cu_batch_cpu},fio)

	np.savetxt('texture_171204_pose3.txt',cano_tex_cpu)
  