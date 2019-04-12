import sys
import torch
import numpy as np
import time
from totaldensify.model.batch_smpl import SmplModel
import totaldensify.data.dataIO as dataIO
from totaldensify.vis.glut_viewer import GLUTviewer
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

	with open('../tests/stage0_171407_haggling_b3_p0.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']
	trans_n += np.array([0,0,50])

	v_all,J_all = smplWrapperCPU(betas_n,thetas_n)
	v_all += trans_n
	J_all += trans_n
	
	
	with open('../tests/stage1_171407_haggling_b3_p0.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']
	trans_n += np.array([0,0,-50])

	
	v_all_new,J_all_new = smplWrapperCPU(betas_n,thetas_n)
	v_all_new += trans_n
	J_all_new += trans_n

	print(v_all_new.shape)
	print(v_all.shape)


	#vc1 = np.loadtxt('./best_texture.txt')
	#vc1 = np.loadtxt('./stage2_texture_framed_interg_100x2.txt')
	vc1 = np.ones((6890,3))*0.5
	vc0 = np.ones((6890,3))*0.75
	
	#vc = np.loadtxt('./stage2_texture.txt')
	
	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'

	n_batch = v_all.shape[0]
	
	f_smpl = smplWrapperCPU.f
	#f_all = f_all[None,:,:].repeat(n_batch,axis=0)

	# vc_all = vc[None,:,:].repeat(n_batch,axis=0)
	mesh_v = []
	
	for i in range(n_batch):
		vt = np.vstack((v_all[i],v_all_new[i]))
		f_c = np.vstack((f_smpl,f_smpl+6890))
		vn = geo_utils.vertices_normals(f_c,vt)

		#vc_tmp = np.ones((6890,3))*0.75
		vc_tmp = np.vstack((vc0[:,::-1],vc1[:,::-1]))

		mesh_v.append({'vt':vt,'vc':vc_tmp,'vn':vn,'f':f_c})
		
	glRender = GLUTviewer(1280, 720)
	glRender.lighting = True
	glRender.loadCalibs(calibFile)

	#use frame 65
	#glRender.load_data([mesh_v[65]],[J_all[65]])
	glRender.load_data(mesh_v,J_all)


	glRender.set_element_status('joints',False)
	glRender.set_element_status('mesh',True)
	glRender.set_element_status('cameras',False)
	glRender.set_element_status('floor',False)
	glRender.start(sys.argv)
