import sys
import torch
import numpy as np
import time
from totaldensify.model.batch_smpl import SmplModel
import totaldensify.data.dataIO as dataIO
from totaldensify.vis.glut_viewer_pcd import GLUTviewer
import cPickle as pickle
import glob
import os.path
import totaldensify.geometry.geometry_process as geo_utils
from totaldensify.utils.config import cfg
import totaldensify.utils.config as cfg_utils

if __name__=='__main__':

	cfg_utils.merge_cfg_from_file('../../config/TotalCapture.yaml')
	cfg_utils.assert_and_infer_cfg()

	model_path = cfg.CAPTURE.MODEL_FEMALE

	smplWrapperCPU = SmplModel(model_path,reg_type='total')
	smplWrapperCPU_male = SmplModel(cfg.CAPTURE.MODEL_MALE,reg_type='total')


	with open('../tests/stage1_new.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']


	v_all,J_all = smplWrapperCPU(betas_n,thetas_n)
	#v_all += trans_n
	#J_all += trans_n
	

	#vc1 = np.loadtxt('./best_texture.txt')
	#vc1 = np.loadtxt('./stage2_texture_framed_interg_100x2.txt')
	vc2 = np.ones((6890,3))*0.85

	vc1 = np.ones((6890,3))*0.85
	vc0 = np.ones((6890,3))*0.85
	
	#vc = np.loadtxt('./stage2_texture.txt')
	
	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'

	n_batch = v_all.shape[0]
	
	f_smpl = smplWrapperCPU.f
	#f_all = f_all[None,:,:].repeat(n_batch,axis=0)

	# vc_all = vc[None,:,:].repeat(n_batch,axis=0)
	mesh_v = []
	
	for i in range(n_batch):
		#vt = np.vstack((v_all[i],v_all_new[i],v_all_2[i]))
		vt = v_all[i]
		f_c = f_smpl
		vn = geo_utils.vertices_normals(f_c,vt)

		vc_tmp = vc0
		mesh_v.append({'vt':vt,'vc':vc_tmp,'vn':vn,'f':f_c})
		
	glRender = GLUTviewer(1280, 720)
	glRender.lighting = True
	glRender.trans = trans_n
	glRender.loadCalibs(calibFile)

	#use frame 65
	#glRender.load_data([mesh_v[65]],[J_all[65]])
	glRender.load_data(mesh_v,J_all)


	glRender.set_element_status('joints',False)
	glRender.set_element_status('mesh',True)
	glRender.set_element_status('cameras',False)
	glRender.set_element_status('floor',False)
	glRender.start(sys.argv)
