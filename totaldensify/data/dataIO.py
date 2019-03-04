import cPickle as pickle
import numpy as np
import numpy.linalg as nalg
import copy
import math
import cv2
import copy
import os
import json

#my own version of hmr data
def parse_hmr_data(hmr_path):
	all_cam = []
	all_trans = []
	all_pose = []
	all_betas = []
	hmr_data_list = []
	with open(hmr_path) as fio:
		hmr_data = pickle.load(fio)

	if not isinstance(hmr_data,list):
		hmr_data_list.append(hmr_data)
	else:
		hmr_data_list = hmr_data


	for hmr in hmr_data_list:
		all_cam.append(hmr['cam'])
		all_trans.append(hmr['trans'])
		all_pose.append(hmr['theta'][3:3+72])
		all_betas.append(hmr['theta'][-10:])

	return {'cam': all_cam, 'trans': all_trans, 'pose': all_pose, 'betas': all_betas}


def normalize_hmr_camera(param_list,cx,cy,f):
	num_ = len(param_list['trans'])

	for idx in range(num_):
		#fix focal length and z
		cam = param_list['cam'][idx]

		param_list['trans'][idx][0] += (cam[1]-cx)/cam[0]*param_list['trans'][idx][2]
		param_list['trans'][idx][1] += (cam[2]-cy)/cam[0]*param_list['trans'][idx][2]
		#change focal length
		param_list['trans'][idx][2] = param_list['trans'][idx][2]/cam[0]*f
	return param_list

def prepare_data_total(root_path,image_path):
	image_name = os.path.basename(image_path)
	image_base = os.path.splitext(image_name)[0]

	img = cv2.imread(image_path)
	img = img[:, :, ::-1]
	op_path = os.path.join(root_path, 'openpose',
						   '{}_keypoints.json'.format(image_base))
	j2d, j2d_w = load_openpose_json(op_path,False)
	
	hmr_path = os.path.join(root_path, 'hmr', '{}.pkl'.format(image_base))
	hmr_data = parse_hmr_data(hmr_path)
	img_width = img.shape[1]
	img_height = img.shape[0]

	hmr_data = normalize_hmr_camera(hmr_data,img_width/2,img_height/2,img_width)

	uv_path = os.path.join(root_path, 'densepose',
						   '{}_IUV.png'.format(image_base))
	uv_img = cv2.imread(uv_path)
	inds_path = os.path.join(root_path,'densepose','{}_INDS.png'.format(image_base))
	inds_img = cv2.imread(inds_path)

	inds_img = solve_inds_map(inds_img,uv_img)
	
	uv_img = uv_img.astype(np.float)
	uv_img = uv_img/256.0

	# bbox = get_bbox(uv_img)
	# uv_img[uv_img[:, :, 0] < 1/256.0, :] = 1
	# uv_img_crop = uv_img[bbox[1]:bbox[3], :, :][:, bbox[0]:bbox[2], :]
	# img_crop = img[bbox[1]:bbox[3], :, :][:, bbox[0]:bbox[2], :]
	#not start with crop and scale

	#from bbox to new cam
	# cam_old = hmr_data['cam']
	# cam_new_cx = cam_old[1] - bbox[0]
	# cam_new_cy = cam_old[2] - bbox[1]

	#normalize all cam

	# j2d[:, 0] = j2d[:, 0] - bbox[0]
	# j2d[:, 1] = j2d[:, 1] - bbox[1]
	

	fit_data = {
		'img':img,
		'uv_img':uv_img,
		'inds_img':inds_img,
		'joints':np.array(j2d),
		'joints_weight':np.array(j2d_w),
		'pose_init':hmr_data['pose'],
		'trans_init':hmr_data['trans'],
		'betas_init':hmr_data['betas'],
		'cam':hmr_data['cam']
	}
	# cam = initialize_camera(cam_old[0], cam_old[0], cam_new_cx, cam_new_cy)
	# cam_old_ch = initialize_camera(cam_old[0], cam_old[0], cam_old[1], cam_old[2])
	#return img,img_crop, uv_img_crop[:, :, ::-1], j2d, j2d_w, hmr_data['pose'], hmr_data['trans'], hmr_data['betas'], cam, cam_old_ch
	#return img,uv_img,j2d,j2d_w,hmr_data['pose'], hmr_data['trans'], hmr_data['betas'], cam
	return fit_data

def load_openpose_json(json_file,only_one=True):

	with open(json_file) as fio:
		cJtr = json.load(fio)
	if only_one:
		cP = cJtr['people'][0]['pose_keypoints_2d']
		cP = np.reshape(cP, (-1, 3))
		weights = np.ones(25)
		weights[cP[:, 2] < 0.3] = 0
		return cP[:, 0:2], weights
	else:
		outJoints = []
		outWeight = []
		for cP in cJtr['people']:
			cP = np.reshape(cP['pose_keypoints_2d'], (-1, 3))
			weights = np.ones(25)
			weights[cP[:, 2] < 0.3] = 0
			outJoints.append(cP[:,0:2])
			outWeight.append(weights)

		return outJoints,outWeight
	
def solve_inds_map(inds_img,uvi_img):
	#def correct_inds_map(inds_img,uvi_img):
	#1. get all existing inds
	inds_img_c0 = inds_img[:,:,0]

	#I_channel of uvi_img
	uv_img_c2 = uvi_img[:,:,-1] 
	
	#temp manually handle run image /databag/net_images/images/run.jpg
	inds_img_c0[inds_img_c0==8] = 14

	#get all existing inds
	inds_set = list(set(inds_img_c0.flatten().tolist()))
	inds_set.sort() #sort it

	#excluding '0'
	inds_set = inds_set[1:]

	mask_img = np.zeros(inds_img_c0.shape,inds_img_c0.dtype)

	#re-arrange idx as 1,2,3 counting numbers 
	for idx,cid in enumerate(inds_set):
		mask_img[inds_img_c0==cid] = idx+1

	#all orignal mask get,now check the consistency between mask img and uvi_img
	uv_img_c2[mask_img>0] = 0
	mask_img[uv_img_c2>0] = len(inds_set)+1

	#stacked as 3-channel grey image, just for consistency
	mask_img = np.dstack((mask_img,mask_img,mask_img))

	return mask_img

def load_dp_vts_2d(root_path,frameId,personId,model='SMPL'):
	view_num = 31
	if(model=='SMPL'):
		all_pts = np.zeros((view_num, 6890,3),np.float32)
	else:
		all_pts = np.zeros((view_num,18540,3),np.float32)
	for view_id in range(view_num):
		c_fileName = '00_{:02d}_{:08d}_{:03d}.txt'.format(view_id, frameId, personId)
		c_filePath = os.path.join(root_path, c_fileName)
		if os.path.isfile(c_filePath):
			c_mat = np.loadtxt(c_filePath)
			c_mat_x = c_mat[:, 1] * 1.5
			c_mat_y = c_mat[:, 2] * 1.5
			c_mat_err = c_mat[:, 0].copy()
			c_mat_err[c_mat[:, 0] > 5e-2] = -1
			all_pts[view_id, :, :] = np.vstack((c_mat_err, c_mat_y, c_mat_x)).T
	return all_pts

def load_dp_vts_3d(root_path,frameId,personId):
	c_fileName = 'pcd_{:08d}_{:03d}.txt'.format(frameId, personId)
	c_filePath = os.path.join(root_path,c_fileName)
	vts_all = np.loadtxt(c_filePath)
	vts_weight = vts_all[:,0]
	vts_weight[vts_weight>=0] = 1
	vts_weight[vts_weight<0] = 0
	vts_3d = vts_all[:,1:]
	return vts_3d,vts_weight
def parse_dome_calibs(calib_file,view_nodes):
	with open(calib_file) as f:
		calibs = json.load(f)

	cameras = calibs['cameras']
	#cameras = rawCalibs['cameras']

	allPanel = [x['panel'] for x in cameras]
	hdCamIndices = [i for i, x in enumerate(allPanel) if x == 0]
	hdCams = [cameras[i] for i in hdCamIndices]

	allNodes = [x['node'] for x in hdCams]

	allIdx = map(lambda x: allNodes.index(x), view_nodes)

	num_view = len(allIdx)

	P_matrices = np.zeros((num_view, 3, 4))

	for i, idx in enumerate(allIdx):
		K = hdCams[idx]['K']
		invR = np.array(hdCams[idx]['R'])
		invT = np.array(hdCams[idx]['t'])
		P_matrices[i, :, :] = np.array(K).dot(np.hstack((invR, invT)))

	return P_matrices