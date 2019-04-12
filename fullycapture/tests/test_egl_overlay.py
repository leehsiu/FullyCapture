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

	with open('../tests/stage1.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']
	#trans_n += np.array([0,0,50])

	v_all,J_all = smplWrapperCPU(betas_n,thetas_n)
	v_all += trans_n
	J_all += trans_n
	
	
	with open('../tests/stage0.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']
	#trans_n += np.array([0,0,-50])
	
	v_all_new,J_all_new = smplWrapperCPU(betas_n,thetas_n)
	v_all_new += trans_n
	J_all_new += trans_n


	vc0 = np.ones((6890,3))*0.25
	vc1 = np.ones((6890,3))*0.75
		
	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'

	n_batch = v_all.shape[0]
	
	f_smpl = smplWrapperCPU.f

	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'
	img_w = 1280.0
	img_h = 720.0

	R_all = np.zeros((31,3,3))
	t_all = np.zeros((31,1,3))
	K_all = np.zeros((31,3,3))
	

	egl = EglRender(int(img_w),int(img_h))
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

	frameId = 0
	camId = 0
	#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))

	while True:
		vt0 = v_all[frameId]
		vt1 = v_all_new[frameId]

		vn0 = geo_utils.vertices_normals(f_smpl,vt0)
		vn1 = geo_utils.vertices_normals(f_smpl,vt1)
		egl.enable_light(True)
		egl.line_width = 0.01
		vis_image0,_ = egl.render_obj(vt0,vc0,vn0,f_smpl,R_all[camId],t_all[camId].T,K_all[camId])
		vis_image1,_ = egl.render_obj(vt1,vc1,vn1,f_smpl,R_all[camId],t_all[camId].T,K_all[camId])

		img_path = '/home/xiul/databag/dome_sptm/171204_pose6/images/00_{:02d}_{:08d}.jpg'.format(camId,frameId+500)
		if os.path.isfile(img_path):
			img = cv2.imread(img_path)
		else:
			img = np.ones((img_h,img_w,3),dtype=np.uint8)
		
		
		img0 = plot_vis.make_overlay(vis_image0[:,:,:3],img,vis_image0[:,:,3])
		img1 = plot_vis.make_overlay(vis_image1[:,:,:3],img,vis_image1[:,:,3])
		img_show0 = np.hstack((img0,img))
		img_show1 = np.hstack((img1,img))
		img_show = np.vstack((img_show0,img_show1))

		img_show = cv2.resize(img_show,(1280,720))
		#out.write(img_show)

		cv2.imshow('img',img_show)
		frameId += 1
		if frameId>= 100:
			frameId = 0

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

