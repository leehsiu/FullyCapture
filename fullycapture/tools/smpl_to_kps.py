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



	#vc1 = np.loadtxt('./best_texture.txt')
	#vc1 = np.loadtxt('./stage2_texture_framed_interg_100x2.txt')
	vc0 = np.ones((6890*3,3))*0.85
	
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


	while True:
		vt0 = v_all[frameId]
		vt1 = v_all_new[frameId]
		vt2 = v_all_2[frameId]
		vt = np.vstack((vt0,vt1,vt2))
		vn0 = geo_utils.vertices_normals(f_c,vt)
		egl.enable_light(True)
		egl.line_width = 0.01
		vis_image0,vis_depth0 = egl.render_obj(vt,vc0,vn0,f_c,R_all[camId],t_all[camId].T,K_all[camId])
		cv2.imshow('img',vis_depth0)
		ky = cv2.waitKey(-1)
		if ky==ord('q'):
			break