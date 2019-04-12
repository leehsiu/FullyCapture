import sys
import cv2
import torch
import numpy as np
import time
from totaldensify.model.batch_smpl import SmplModel
import totaldensify.data.dataIO as dataIO
import cPickle as pickle
import glob
import os.path
import totaldensify.geometry.geometry_process as geo_utils
from totaldensify.utils.config import cfg
import totaldensify.utils.config as cfg_utils
from totaldensify.vis.egl_render import EglRender
if __name__=='__main__':

	cfg_utils.merge_cfg_from_file('../../config/TotalCapture.yaml')
	cfg_utils.assert_and_infer_cfg()

	model_path = cfg.CAPTURE.MODEL_FEMALE
	smplWrapperCPU = SmplModel(model_path,reg_type='total')
	smplWrapperCPU_male = SmplModel(cfg.CAPTURE.MODEL_MALE,reg_type='total')
	with open('../tests/stage0_171407_haggling_b3_p0.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']
	v0,_ = smplWrapperCPU(betas_n,thetas_n)
	v0 += trans_n
	
	with open('../tests/stage0_171407_haggling_b3_p1.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']

	v1,_ = smplWrapperCPU_male(betas_n,thetas_n)
	v1 += trans_n

	with open('../tests/stage0_171407_haggling_b3_p2.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']

	v2,_ = smplWrapperCPU(betas_n,thetas_n)
	v2 += trans_n

	vc0 = np.ones((6890,3))*0.25
	vc1 = np.ones((6890,3))*0.5
	vc2 = np.ones((6890,3))*0.75

	n_view_batch = 31
	calibFile = '/home/xiul/databag/dome_sptm/170407_haggling_b3/calibration_170407_haggling_b3.json'
	import json
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

	for camId in range(n_view_batch):
		R = np.array(hdCams[camId]['R'])
		t = np.array(hdCams[camId]['t']).T
		K = np.array(hdCams[camId]['K'])*1280.0/1920.0
		R_all[camId] = R
		t_all[camId] = t
		K_all[camId] = K

	n_batch = v0.shape[0]

	
	f_smpl = smplWrapperCPU.f
	f_c = np.vstack((f_smpl,f_smpl+6890,f_smpl+6890*2))

		#vc_tmp = np.ones((6890,3))*0.75
	vc_tmp = np.vstack((vc0,vc1,vc2))
	mesh_v = []

	mainRender = EglRender(1280,720)
	mainRender.lighting = False

	inds_list = [64,127,191]

	for frameId in range(n_batch):
		for viewId in range(n_view_batch):
			inds_GT = np.zeros((720,1280,3),dtype=np.uint8)
			vt_all = np.vstack((v0[frameId],v1[frameId],v2[frameId]))
			inds,_ = mainRender.render_obj(vt_all,vc_tmp,None,f_c,R_all[viewId],t_all[viewId].T,K_all[viewId])
			inds = inds[:,:,:3]
			inds_dp = cv2.imread('/home/xiul/databag/dome_sptm/170407_haggling_b3/densepose/00_{:02d}_{:08d}_INDS.png'.format(viewId,frameId+3500))
		
			for mid,cid in enumerate(inds_list):
				c_mask = np.zeros(inds.shape,dtype=np.uint8)
				c_mask[inds[:,:,0]==cid,:] = 1
				inds_dp_masked = inds_dp*c_mask
				inds_dp_masked0 = inds_dp_masked[:,:,0]
				inds_dp_list = inds_dp_masked0.flatten().tolist()
				inds_set = list(set(inds_dp_list)- set([0]))
				all_count = []
				for iid in inds_set:
					all_count.append(inds_dp_list.count(iid))
				if len(all_count)>0:
					max_id = np.argmax(all_count)
					associate_id = inds_set[max_id]
					inds_GT[inds_dp[:,:,0]==associate_id,:] = cid

			out_name = '/home/xiul/databag/dome_sptm/170407_haggling_b3/densepose/00_{:02d}_{:08d}_INDS_GT.png'.format(viewId,frameId+3500)
			#cv2.imshow('inds_gt',inds_dp*10)	
			cv2.imwrite(out_name,inds_GT)
			print('{}:{}'.format(frameId,viewId))