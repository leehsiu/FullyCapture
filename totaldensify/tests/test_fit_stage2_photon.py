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

	model_path = cfg.CAPTURE.MODEL
	capUtils = capture_utils.StdCapture(model_path,reg_type='total')

	
	with open('../tests/stage1.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']

	v_all,J_all = capUtils.smplWrapperCPU(betas_n,thetas_n)
	v_all += trans_n
	J_all += trans_n
	
	#use frame 0


	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'
	img_w = 1280.0
	img_h = 720.0
	egl = EglRender(int(img_w),int(img_h))


	with open(calibFile,'r') as fio:
		rawCalibs = json.load(fio)
	cameras = rawCalibs['cameras']
	allPanel = map(lambda x:x['panel'],cameras)
	hdCamIndices = [i for i,x in enumerate(allPanel) if x==0]
	hdCams = [cameras[i] for i in hdCamIndices]

	n_view_batch = 31

	#step.0 get the visibility map
	vis_mask = np.zeros((n_view_batch,6890,1),dtype=np.float32)
	vis_img = []
	vt = v_all[0]
	#vc = np.ones((6890,3),dtype=np.float32)
	
	vc_dp = scipy.io.loadmat('/home/xiul/workspace/PanopticDome/matlab/SMPL_DP.mat')['SMPL_IUV'][:,1:]
	vc_dp[:,2] = vc_dp[:,2]/30.0
	vc_dp_char_float = (vc_dp*255)


	R_all = np.zeros((n_view_batch,3,3))
	t_all = np.zeros((n_view_batch,1,3))
	K_all = np.zeros((n_view_batch,3,3))
	f = capUtils.smplWrapperCPU.f
	for camId in range(n_view_batch):
		R = np.array(hdCams[camId]['R'])
		t = np.array(hdCams[camId]['t']).T
		K = np.array(hdCams[camId]['K'])*1280.0/1920.0
		R_all[camId] = R
		t_all[camId] = t
		K_all[camId] = K
		vn = geo_utils.vertices_normals(f,vt)
		vis_image,vis_depth = egl.render_obj(vt,vc_dp,vn,f,R,t.T,K)
		vis_img.append(vis_depth)

	vis_img = np.array(vis_img)
	vis_img_cu = torch.from_numpy(vis_img[:,:,:,None].transpose(0,3,1,2).astype(np.float32)).cuda()


	vt_cu = torch.tensor(vt[None,:,:].repeat(n_view_batch,axis=0),dtype=torch.float32).cuda()
	K_cu = torch.tensor(K_all,dtype=torch.float32).cuda()
	R_cu = torch.tensor(R_all,dtype=torch.float32).cuda()
	t_cu = torch.tensor(t_all,dtype=torch.float32).cuda()


	vt_2d_cu = geo_utils.projection(vt_cu,K_cu,R_cu,t_cu,img_w,img_h)
	vt_2d_grid = vt_2d_cu[:,:,None,:2]

	depth_2d = F.grid_sample(vis_img_cu,vt_2d_grid,mode='nearest')
	depth_2d = depth_2d.cpu().numpy().transpose(0,2,1,3)[:,:,:,0]
	vt_2d_cpu = vt_2d_cu.cpu().numpy()

	vis_diff = nlg.norm(depth_2d - vt_2d_cpu[:,:,2][:,:,None],ord=np.inf,axis=2)[:,:,None]
	vis_mask[vis_diff<1.0] = 1
	#the visibiliy mask

	#try the render here.
	# fig,ax = plt.subplots(1)


	cano_tex = torch.zeros(1,3,6890,1,requires_grad=True,device='cuda',dtype=torch.float32).cuda()
	vis_mask_cu = torch.tensor(vis_mask[:,None,:,:].repeat(3,axis=1),dtype=torch.float32).cuda()
	
	#read images
	img_target = []
	for view_id in range(n_view_batch):
		img_path = '/home/xiul/databag/dome_sptm/171204_pose6/images/00_{:02d}_{:08d}.jpg'.format(view_id,500)
		if os.path.isfile(img_path):
			img0 = cv2.imread(img_path).astype(np.float32)
		else:
			img0 = np.zeros((720,1280,3),dtype=np.float32)
			vis_mask_cu[view_id] = 0
		img_target.append(img0)
	img_target = np.array(img_target)
	
	img_target_cu = torch.tensor(img_target.transpose(0,3,1,2)/255.0,dtype=torch.float32).cuda()

	l2loss = torch.nn.MSELoss().cuda()

	optimizer = torch.optim.Adam([cano_tex],lr=0.01)

	for i_iter in range(100):
		cano_tex_batch = cano_tex.repeat(n_view_batch,1,1,1)
		vt_2d_cu = geo_utils.projection(vt_cu,K_cu,R_cu,t_cu,img_w,img_h)
		vt_2d_grid = vt_2d_cu[:,:,None,:2]
		rep_color_cu = F.grid_sample(img_target_cu,vt_2d_grid)
		
		loss = l2loss(rep_color_cu*vis_mask_cu,cano_tex_batch*vis_mask_cu)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('loss is {}'.format(loss.item()))
	

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
	
	np.savetxt('stage2_texture.txt',cano_tex_cpu)
  