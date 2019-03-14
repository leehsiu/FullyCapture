import cPickle as pickle
import numpy as np
import numpy.linalg as nalg
import copy
import math
import cv2
import copy
import os
import json
import glob


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

	return fit_data

def prepare_data_total_monocular(root_path):

	print(root_path)
	#image_names = glob.glob(root_path+'images')
	image_names = ['/home/xiul/databag/dslr_dance/images/dslr_dance1_{:012d}_rendered.png'.format(i) for i in range(100)]
	feat_names = ['/home/xiul/databag/dslr_dance/dense_feature/dslr_dance1_{:012d}_rendered.txt'.format(i) for i in range(100)]
	#image_names = image_names.sort()
	#print(image_names)
	joints = []
	joints_weight = []

	#verts = []
	imgs = []
	hmr_cam = []
	hmr_trans = []
	hmr_pose = []
	hmr_betas = []



	for image_path in image_names:
		image_base = os.path.basename(image_path)
		image_base = os.path.splitext(image_base)[0]
		img = cv2.imread(image_path)
		img = img[:, :, ::-1]
		op_path = os.path.join(root_path, 'openpose',
						   '{}_keypoints.json'.format(image_base))
		j2d, j2d_w = load_openpose_json(op_path,True)
	
		hmr_path = os.path.join(root_path, 'hmr', '{}.pkl'.format(image_base))
		hmr_data = parse_hmr_data(hmr_path)
		
		imgs.append(img)
		joints.append(j2d)
		joints_weight.append(j2d_w)

		hmr_cam.append(hmr_data['cam'][0])
		hmr_trans.append(hmr_data['trans'][0])
		hmr_pose.append(hmr_data['pose'][0])
		hmr_betas.append(hmr_data['betas'][0])
	#return {'cam': all_cam, 'trans': all_trans, 'pose': all_pose, 'betas': all_betas}
	# img = cv2.imread(image_names[0])
	
	# imgs.append(img)
	hmr_all = {'cam':hmr_cam,'trans':hmr_trans,'pose':hmr_pose,'betas':hmr_betas}
	
	img_width = imgs[0].shape[1]
	img_height = imgs[0].shape[0]

	hmr_all = normalize_hmr_camera(hmr_all,img_width/2,img_height/2,img_width)

	verts = load_dp_vts_2d_monocular(feat_names)
	fit_data = {
		'img':imgs,
		'joints':np.array(joints),
		'joints_weight':np.array(joints_weight),
		'verts':verts,
		'pose':hmr_all['pose'],
		'trans':hmr_all['trans'],
		'betas':hmr_all['betas'],
		'cam':hmr_all['cam']
	}
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


def load_dp_vts_2d_monocular(file_lists,model='SMPL'):
	n_batch = len(file_lists)

	if(model=='SMPL'):
		all_pts = np.zeros((n_batch, 6890,3),np.float32)
	else:
		all_pts = np.zeros((n_batch,18540,3),np.float32)
	for idx,cfile in enumerate(file_lists):
		#c_filePath = os.path.join(root_path, c_fileName)
		c_mat = np.loadtxt(cfile)
		c_mat_x = c_mat[:, 1]
		c_mat_y = c_mat[:, 2]
		c_mat_err = c_mat[:, 0].copy()
		c_mat_err[c_mat[:, 0] > 5e-2] = -1
		all_pts[idx, :, :] = np.vstack((c_mat_err, c_mat_y, c_mat_x)).T
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


def load_dome_calibs(calib_file,view_nodes):
	with open(calib_file) as f:
		calibs = json.load(f)

	#calibs = json.load(f)
	cameras = calibs['cameras']

	allPanel = [x['panel'] for x in cameras]
	hdCamIndices = [i for i, x in enumerate(allPanel) if x == 0]
	hdCams = [cameras[i] for i in hdCamIndices]

	allNodes = [x['node'] for x in hdCams]

	allIdx = map(lambda x: allNodes.index(x), view_nodes)

	num_view = len(allIdx)
	K_matrices = np.zeros((num_view,3,3))
	R_matrices = np.zeros((num_view,3,3))
	t_matrices = np.zeros((num_view,1,3))
	for i, idx in enumerate(allIdx):
		K = hdCams[idx]['K']
		invR = np.array(hdCams[idx]['R'])
		invT = np.array(hdCams[idx]['t'])
		K_matrices[i,:,:] = np.array(K)
		R_matrices[i,:,:] = np.array(invR)
		t_matrices[i,:,:] = np.array(invT).T

		#P_matrices[i, :, :] = np.array(K).dot(np.hstack((invR, invT)))
	return K_matrices,R_matrices,t_matrices
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

def load_dome_images_wrt_frame(root_path,frame_id,view_ids):
	out_images = []
	for vid in view_ids:
		file_name = os.path.join(root_path,'00_{:02d}_{:08d}.jpg'.format(vid,frame_id))
		print(file_name)
		img = cv2.imread(file_name)
		if img is None:
			print('got empty image named {}-{}'.format(frame_id,vid))
		out_images.append(img)
	return out_images


def load_total_joints_3d(file_path):
	with open(file_path) as fio:
		dd = json.load(fio)
	return dd
	