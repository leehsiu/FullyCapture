# Discuss. DCT-resampling in fitting loop, or firstly DCT-resample, then fitting.

# direct use DCT-basis to resample the joints.

# 0. Nearest search. 
# 1. DCT-resample and fitting.
# 2. Photometric
import glob
import os.path
import sys
from PyQt5 import QtWidgets
from fullycapture.app.pyqtDomePlayer import PanopticStudioPlayer
import json
import scipy.signal
import numpy as np

if __name__=='__main__':
	#load data.

		
	app = QtWidgets.QApplication(sys.argv)
	m_player = PanopticStudioPlayer()
	m_player.dataroot = '/media/internal/domedb'
	m_player.seqname = '171204_pose6'


	seqname = '171204_pose6'
	joint_files = glob.glob(os.path.join('/media/internal/domedb',seqname,'hdPose3d_total/*.json'))
	joint_files.sort()
	
	joint_mat = []
	for c_file in joint_files[:1024]:
		with open(c_file) as fio:
			c_str = json.load(fio)
		c_joint = c_str[0]['joints25']
		joint_mat.append(c_joint)
	joint_mat = np.array(joint_mat)


	#simple median fit with kernel_size = 5
	joint_mat_filter = []
	for col in joint_mat.T:
		joint_mat_filter.append(scipy.signal.medfilt(col,kernel_size=5))
	joint_mat_filter = np.array(joint_mat_filter).T


	def custom_load_cameras(dataroot,seqname):
		global_root = '/media/posefs0c/panoptic'
		calib_file = os.path.join(global_root,seqname,'calibration_{}.json'.format(seqname))
		with open(calib_file) as f:
			calib_json_str = json.load(f)
		cameras = calib_json_str['cameras']
		allPanel = map(lambda x:x['panel'],cameras)
		hdCamIndices = [i for i,x in enumerate(allPanel) if x==0]
		cam_list = [cameras[i] for i in hdCamIndices]
		return cam_list


	def custom_load_timeline(dataroot,seqname):
		return joint_mat_filter.shape[0]

	def custom_load_joints(dataroot,seqname,frame_id):
		jtr1 = joint_mat[frame_id]
		jtr2 = joint_mat_filter[frame_id]
		jtr1_25 = np.reshape(jtr1,(-1,4))
		jtr2_25 = np.reshape(jtr2,(-1,4))
		jtr2_25[:,3] = jtr1_25[:,3]
		jtr2_25[:,2] += 150
		joints_total = []
		c_joint1 = {'id':0,
					'joints25':jtr1_25,
					'right_hand':np.zeros((21,4)),
					'left_hand':np.zeros((21,4)),
					'face70':np.zeros((70,4))
					}
		c_joint2 = {'id':1,
					'joints25':jtr2_25,
					'right_hand':np.zeros((21,4)),
					'left_hand':np.zeros((21,4)),
					'face70':np.zeros((70,4))}

		joints_total.append(c_joint1)
		joints_total.append(c_joint2)

		return joints_total

	def custom_load_mesh(dataroot,seqname,frame_id):		
		#dataroot is '/media/posefs3b/Users
		return None

	m_player.custom_load_cameras = custom_load_cameras
	m_player.custom_load_timeline = custom_load_timeline
	m_player.custom_load_joints = custom_load_joints
	m_player.custom_load_mesh = custom_load_mesh
	m_player.reset_widget()
	m_player.loadDB_core()
	m_player.show()

	sys.exit(app.exec_())  

