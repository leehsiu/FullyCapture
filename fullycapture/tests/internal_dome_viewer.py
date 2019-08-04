import sys
from PyQt5 import QtWidgets
from fullycapture.app.pyqtDomePlayer import PanopticStudioPlayer
import os.path
import json
import cPickle as pickle
import glob
import numpy as np
import fullycapture.geometry.geometry_process as geo_utils
from fullycapture.model.batch_adam import TotalModel
import time
if __name__=='__main__':

	app = QtWidgets.QApplication(sys.argv)
	m_player = PanopticStudioPlayer()
	m_player.dataroot = '/media/internal/domedb'
	m_player.seqname = '171204_pose6'

	joints_file = glob.glob(os.path.join(m_player.dataroot,m_player.seqname,'hdPose3d_total','*.json'))
	joints_file.sort()


	adamPath = '/home/xiul/workspace/FullyCapture/models/adamModel_with_coco25_reg.pkl'
	totalModelWrapper = TotalModel(pkl_path=adamPath)



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
		frame_num = len(joints_file)
		return frame_num

	def custom_load_joints(dataroot,seqname,frame_id):
		coco25_file = joints_file[frame_id]
		joints_total = []
		with open(coco25_file) as fio:
			jsonStr = json.load(fio)
		for entry  in jsonStr:
			cid = entry['id']

			joints25 = np.reshape(entry['joints25'],(-1,4))
			c_joint = {'id':cid,
					   'joints25':joints25,
					   'right_hand':np.zeros((21,4)),
					   'left_hand':np.zeros((21,4)),
					   'face70':np.zeros((70,4))
						}
			joints_total.append(c_joint)
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

