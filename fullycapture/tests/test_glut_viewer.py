import sys
import torch
import numpy as np
import time
from fullycapture.model.batch_smpl import SmplModel
import fullycapture.data.dataIO as dataIO
from fullycapture.vis.glut_viewer import GLUTviewer
import fullycapture.geometry.geometry_process as geo_utils
from fullycapture.utils.config import cfg
import fullycapture.utils.config as cfg_utils
import pickle
import glob
import os.path

if __name__=='__main__':

	cfg_utils.merge_cfg_from_file('../../config/TotalCapture.yaml')
	cfg_utils.assert_and_infer_cfg()

	model_path = cfg.CAPTURE.MODEL_NEUTRAL
	smplWrapperCPU = SmplModel(model_path,reg_type='total')

	mesh_path = '/media/internal/domedb/171204_pose6/model_stage0_proposal'
	mesh_files = glob.glob(os.path.join(mesh_path,'*.pkl'))
	mesh_files.sort()

	joints_path = '/media/internal/domedb/171204_pose6/hdPose3d_total'
	joints_files = glob.glob(os.path.join(joints_path,'*.json'))
	joints_files.sort()

	frame_num = len(mesh_files)


	def mesh_load_func(frameid):
		pkl_file = mesh_files[frameid]
		with open(pkl_file,'rb') as fio:
			dd = pickle.load(fio)
		theta = dd['theta']
		beta = dd['beta']
		trans = dd['trans']
		trans += [0,0,50]
		ids = dd['ids']
		vt,_ = smplWrapperCPU(beta,theta)
		vt += trans

		vt = vt.reshape(-1,3)
		vf = []
		vc = []
		for ip,id_c in enumerate(ids):
			vf.append(smplWrapperCPU.f + ip*smplWrapperCPU.size[0])
			c_level = (id_c % 10)+1
			c_level = c_level*0.05
			vc.append(np.ones(smplWrapperCPU.size)-c_level)
		vf = np.concatenate(vf)
		vc = np.concatenate(vc)
		vn = geo_utils.vertices_normals(f=vf,v=vt)
		return vt,vc,vn,vf


	def joints_load_func(frameid):
		cfile = joints_files[frameid]
		dds = dataIO.load_total_joints_3d(cfile)
		joints_total = []
		for dd in dds:
			j3d_t = np.zeros((65,4))
			j3d_t[0:25,:] = np.reshape(dd['joints25'],(-1,4))
			j3d_t[25:45,:] = np.reshape(dd['right_hand'],(-1,4))[1:,:]
			j3d_t[45:,:] = np.reshape(dd['left_hand'],(-1,4))[1:,:]
			joints_total.append(j3d_t)
		joints_total = np.array(joints_total)
		return joints_total
		
	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'
	glRender = GLUTviewer(1280, 720)
	glRender.lighting = True
	glRender.loadCalibs(calibFile)


	glRender.set_load_func('mesh',mesh_load_func)
	glRender.set_load_func('joints',joints_load_func)
	glRender.set_frame_num(frame_num)
	glRender.set_element_status('joints',True)
	glRender.set_element_status('mesh',True)
	glRender.set_element_status('cameras',False)
	glRender.set_element_status('floor',True)
	glRender.start(sys.argv)
