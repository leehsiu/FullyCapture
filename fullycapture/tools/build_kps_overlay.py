import sys
import torch
import numpy as np
import time
from totaldensify.model.batch_smpl import SmplModel
import totaldensify.data.dataIO as dataIO
from totaldensify.vis.egl_render import EglRender
import cPickle as pickle
import glob
import os.path
import totaldensify.geometry.geometry_process as geo_utils
from totaldensify.utils.config import cfg
import totaldensify.utils.config as cfg_utils
import json
import totaldensify.vis.plot_vis as plot_vis
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as color_map
import glob
import scipy.io

op107_smpl_id = scipy.io.loadmat('/home/xiul/workspace/PanopticDome/iccv2019_matlab/op107_smpl.mat')['op107_smpl'].astype(np.int)

smpl_iuv_path = '../../models/smpl_iuv.txt'
smpl_iuv = np.loadtxt(smpl_iuv_path)
smpl_iuv[:,0] = smpl_iuv[:,0]/255.0
#smpl_iuv = smpl_iuv[:,[1,2,0]]
#print(smpl_iuv)


if __name__=='__main__':

	cfg_utils.merge_cfg_from_file('../../config/TotalCapture.yaml')
	cfg_utils.assert_and_infer_cfg()

	model_path = cfg.CAPTURE.MODEL_FEMALE

	smplWrapperCPU = SmplModel(model_path,reg_type='total')
	smplWrapperCPU_male = SmplModel(cfg.CAPTURE.MODEL_MALE,reg_type='total')


	with open('../tests/stage1_171407_haggling_b3_p0.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']


	v_all,J_all = smplWrapperCPU(betas_n,thetas_n)
	v_all += trans_n
	J_all += trans_n
	
	
	with open('../tests/stage1_171407_haggling_b3_p1.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']

	
	v_all_new,J_all_new = smplWrapperCPU_male(betas_n,thetas_n)
	v_all_new += trans_n
	J_all_new += trans_n


	with open('../tests/stage1_171407_haggling_b3_p2.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']

	
	v_all_2,J_all_2 = smplWrapperCPU(betas_n,thetas_n)
	v_all_2 += trans_n
	J_all_2 += trans_n

	vc0 = np.vstack((smpl_iuv,smpl_iuv,smpl_iuv))
	#vc = np.loadtxt('./stage2_texture.txt')
	
	calibFile = '/home/xiul/databag/dome_sptm/170407_haggling_b3/calibration_170407_haggling_b3.json'

	n_batch = v_all.shape[0]
	
	f_smpl = smplWrapperCPU.f
	#f_all = f_all[None,:,:].repeat(n_batch,axis=0)

	# vc_all =
	img_w = 1280
	img_h = 720

	egl = EglRender(int(img_w),int(img_h))
	R_all = np.zeros((31,3,3))
	t_all = np.zeros((31,1,3))
	K_all = np.zeros((31,3,3))
	


	with open(calibFile,'r') as fio:
		rawCalibs = json.load(fio)
	cameras = rawCalibs['cameras']
	allPanel = map(lambda x:x['panel'],cameras)
	hdCamIndices = [i for i,x in enumerate(allPanel) if x==0]
	hdCams = [cameras[i] for i in hdCamIndices]

	for camId in range(31):
		R = np.array(hdCams[camId]['R'])
		t = np.array(hdCams[camId]['t']).T
		K = np.array(hdCams[camId]['K'])*1280.0/1920.0

		R_all[camId] = R
		t_all[camId] = t
		K_all[camId] = K

	frameId = 24
	camId = 0
	#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))
	f_c = np.vstack((f_smpl,f_smpl+6890,f_smpl+6890*2))

	egl.set_flat(True)
	while True:
		vt0 = v_all[frameId]
		vt1 = v_all_new[frameId]
		vt2 = v_all_2[frameId]
		vt = np.vstack((vt0,vt1,vt2))

		vn0 = geo_utils.vertices_normals(f_c,vt)
		egl.enable_light(False)
		egl.line_width = 0.01
		vis_image0,_ = egl.render_obj(vt,vc0,vn0,f_c,R_all[camId],t_all[camId].T,K_all[camId])
		img_path = '/home/xiul/databag/dome_sptm/170407_haggling_b3/images/00_{:02d}_{:08d}.jpg'.format(camId,frameId+3500)
		if os.path.isfile(img_path):
			img = cv2.imread(img_path)
		else:
			img = np.ones((img_h,img_w,3),dtype=np.uint8)
		
		
		print(camId)
		img0 = vis_image0
		#img0 = plot_vis.make_overlay(vis_image0[:,:,:3],img,vis_image0[:,:,3])

		cv2.imshow('img',img0[:,:,:])

		ky = cv2.waitKey(1)
		if ky==ord('q'):
			break
		elif ky==ord('a'):
			camId -= 1
		elif ky==ord('d'):
			camId +=1
		if camId<0:
			camId = 0
		if camId>30:
			camId=30
	#out.release()

