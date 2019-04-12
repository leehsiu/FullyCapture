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
fig,ax = plt.subplots(4,1)
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

	n_frame_batch = betas_n.shape[0]

	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'
	img_w = 1280.0
	img_h = 720.0
	egl = EglRender(int(img_w),int(img_h))
	

	#load the v_pattern.
	v_pattern = np.loadtxt('../../models/smpl_v_pattern.txt').astype(np.int32)

	# f1----fn [v0] , f1-----fn  [v1] f1 ------fn ...


	with open(calibFile,'r') as fio:
		rawCalibs = json.load(fio)
	cameras = rawCalibs['cameras']
	allPanel = map(lambda x:x['panel'],cameras)
	hdCamIndices = [i for i,x in enumerate(allPanel) if x==0]
	hdCams = [cameras[i] for i in hdCamIndices]
	#preload camera matrixs
	R_all = np.zeros((n_view_batch,3,3))
	t_all = np.zeros((n_view_batch,1,3))
	K_all = np.zeros((n_view_batch,3,3))
	smpl_f = capUtils.smplFemaleCpu.f
	
	for camId in range(n_view_batch):
		R = np.array(hdCams[camId]['R'])
		t = np.array(hdCams[camId]['t']).T
		K = np.array(hdCams[camId]['K'])*1280.0/1920.0
		R_all[camId] = R
		t_all[camId] = t
		K_all[camId] = K

	K_cu = torch.tensor(K_all,dtype=torch.float32).cuda()
	R_cu = torch.tensor(R_all,dtype=torch.float32).cuda()
	t_cu = torch.tensor(t_all,dtype=torch.float32).cuda()


	tt_error = np.zeros((n_frame_batch,6890),dtype=np.float32)


	vc_dummpy = np.zeros((6890,3),dtype=np.float32)
	for frameid in range(n_frame_batch):
		vis_depth_full = []
		vis_mask = np.zeros((n_view_batch,6890,1),dtype=np.float32)
		vt = v_all[frameid]
		vn = geo_utils.vertices_normals(smpl_f,vt)
		for camId in range(n_view_batch):
			_,vis_depth = egl.render_obj(vt,vc_dummpy,vn,smpl_f,R_all[camId],t_all[camId].T,K_all[camId])
			vis_depth_full.append(vis_depth)
		vis_depth_full = np.array(vis_depth_full)
		vis_depth_cu = torch.from_numpy(vis_depth_full[:,:,:,None].transpose(0,3,1,2).astype(np.float32)).cuda()
		vt_cu = torch.tensor(vt[None,:,:].repeat(n_view_batch,axis=0),dtype=torch.float32).cuda()
		vt_2d_cu = geo_utils.projection_cuda(vt_cu,K_cu,R_cu,t_cu,img_w,img_h)
		vt_2d_cu_grid = vt_2d_cu[:,:,:2]
		vt_2d_cu_grid = vt_2d_cu_grid[:,v_pattern,:]
		vt_2d = vt_2d_cu.cpu().numpy()
		depth_sample=  F.grid_sample(vis_depth_cu,vt_2d_cu_grid,mode='nearest')
		depth_2d_render = depth_sample.cpu().numpy().transpose(0,2,3,1)[:,:,:,0]
		depth_2d_proj = vt_2d[:,v_pattern,2]
		vis_diff = nlg.norm(depth_2d_proj - depth_2d_render,ord=np.inf,axis=2)
		vis_mask[vis_diff<5e-1] = 1


		img_target = []
		for view_id in range(n_view_batch):
			img_path = '/home/xiul/databag/dome_sptm/171204_pose6/images/00_{:02d}_{:08d}.jpg'.format(view_id,frameid+500)
			if os.path.isfile(img_path):
				img0 = cv2.imread(img_path).astype(np.float32)
			else:
				img0 = np.zeros((720,1280,3),dtype=np.float32)
				vis_mask[view_id,:] = 0
			img_target.append(img0)
		img_target = np.array(img_target)
		img_target_cu = torch.tensor(img_target.transpose(0,3,1,2)/255.0,dtype=torch.float32).cuda()
		img_sample=  F.grid_sample(img_target_cu,vt_2d_cu_grid)
		img_sample_render = img_sample.cpu().numpy().transpose(0,2,3,1)		
		vis_mask_num = np.sum(vis_mask,axis=0)
		vis_mask[:,vis_mask_num<4] = 0

		sdd_error = np.ones((6890,1))*-1
		vert_color = np.zeros((6890,3),dtype=np.float32)
		for vertid in range(6890):
			c_vert_mask = vis_mask[:,vertid,0]

			c_vert_color = img_sample_render[c_vert_mask>0,vertid,:,:]
			n_vert_mask = c_vert_color.shape[0]
			c_view, = np.where(c_vert_mask>0)


			if n_vert_mask>4:
				var_r = np.var(c_vert_color[:,:,0],axis=0)
				var_g = np.var(c_vert_color[:,:,1],axis=0)
				var_b = np.var(c_vert_color[:,:,2],axis=0)
				var_total = np.mean(var_r + var_g + var_b)
				sdd_error[vertid] = var_total

				if vertid == 2801:
					
					for idx,ii in enumerate(c_view[:4]):
						v_pos = vt_2d[ii,vertid,:2]
						v_pos_x = v_pos[0]*640+640
						v_pos_y = v_pos[1]*360+360

						x_ = int(v_pos[0]*640+640 - 30)
						y_ = int(v_pos[1]*360+360 - 30)
						img_bbox = img_target[ii][y_:y_+60,x_:x_+60,::-1]/255.0
						v_2d_x = vt_2d[ii,v_pattern[vertid],0]*640+640
						v_2d_y = vt_2d[ii,v_pattern[vertid],1]*360+360
						v_2d_x = (v_2d_x - v_pos_x)*2.5+v_pos_x
						v_2d_y = (v_2d_y - v_pos_y)*2.5+v_pos_y

						v_2d_x = v_2d_x - x_
						v_2d_y = v_2d_y - y_
						
						
						ax[idx].clear()
						ax[idx].imshow(img_bbox)
						c_patten = v_pattern[vertid]
						ax[idx].scatter(v_2d_x,v_2d_y,s=0.5,c='y')
						ax[idx].scatter(v_pos_x-x_,v_pos_y-y_,s=2,c='r',marker='*')
						ax[idx].set_xticks(())
						ax[idx].set_yticks(())
						
						
					plt.draw()
					plt.pause(0.01)
					raw_input()
			else:
				sdd_error[vertid] = 1e8
			


		tt_error[frameid,:] = sdd_error.ravel()


		#visualization 


		# for viewId in range(n_view_batch):
		# 	ax.clear()
		# 	vis_id = np.where(np.logical_and(vis_mask_num>=4,vis_mask[viewId,:]>0))
		# 	ax.scatter(vt_2d[viewId,vis_id,0],vt_2d[viewId,vis_id,1])
		# 	#ax.plot(vis_mask_num)
		# 	plt.draw()
		# 	plt.pause(0.01)
		# 	raw_input()