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

	model_path = cfg.CAPTURE.MODEL

	smplWrapperCPU = SmplModel(model_path,reg_type='total')

	with open('../tests/stage1.pkl','r') as fio:
		dd = pickle.load(fio)
	betas_n = dd['betas']
	thetas_n = dd['pose']
	trans_n = dd['trans']

	v_all,J_all = smplWrapperCPU(betas_n,thetas_n)
	v_all += trans_n
	J_all += trans_n

	vc = np.loadtxt('./stage2_texture.txt')
	

	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'

	n_batch = v_all.shape[0]
	
	f_all = smplWrapperCPU.f
	f_all = f_all[None,:,:].repeat(n_batch,axis=0)
	vc_all = vc[None,:,:].repeat(n_batch,axis=0)
	mesh_v = []
	
	for i in range(n_batch):
		vn = geo_utils.vertices_normals(f_all[i],v_all[i])
		mesh_v.append({'vt':v_all[i],'vc':vc_all[i][:,::-1],'vn':vn,'f':f_all[i]})

	glRender = GLUTviewer(1280, 720)
	glRender.loadCalibs(calibFile)
	glRender.load_data(mesh_v,J_all[:,:25,:])
	glRender.set_element_status('joints',False)
	glRender.start(sys.argv)
