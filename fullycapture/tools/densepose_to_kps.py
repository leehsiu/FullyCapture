import totaldensify.data.dp_methods as dp_utils
import os
import os.path
import glob
import cv2
import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt

subdivs = scipy.io.loadmat('/home/xiul/workspace/PanopticDome/models/SMPL_subdiv.mat')
kps107 = scipy.io.loadmat('/home/xiul/workspace/PanopticDome/iccv2019_matlab/op107_kps')['op107_kps'].astype(np.int).ravel()


sI = subdivs['Part_ID_subdiv'][kps107]
sU = subdivs['U_subdiv'][kps107]
sV = subdivs['V_subdiv'][kps107]


uv_table = np.hstack((sI,sU,sV))
print(uv_table.shape)

if __name__=='__main__':
	
	rootpath = '/home/xiul/databag/dome_sptm/170407_haggling_b3' 
	outpath = '/home/xiul/databag/dome_sptm/170407_haggling_b3/sample'
	if not os.path.isdir(outpath):
		os.mkdir(outpath)

	pklFiles = glob.glob(os.path.join(rootpath,'sample','*_IUV_GT.png'))
	pklFiles.sort()
	for cpkl in pklFiles:
		cpklName = os.path.basename(cpkl)
		cpklName = os.path.splitext(cpklName)[0]
		inds_path = os.path.join(rootpath,'sample','{}_INDS.png'.format(cpklName[:-7]))

		t0 = time.time()
		uv_img = cv2.imread(cpkl)
		inds_img = cv2.imread(inds_path)
		inds_img0 = inds_img[:,:,0]
		inds_set = list(set(inds_img0.flatten().tolist()) - set([0]))
		for cid,inds in enumerate(inds_set):
			uv_mask = np.zeros(uv_img.shape,dtype=np.uint8)
			uv_mask[inds_img[:,:,0]==inds,:] = 1
			uv_img_masked = uv_img * uv_mask
			res =dp_utils.dp_uvi_to_verts(uv_img_masked,uv_table,[])
			outfilepath = os.path.join(outpath,'op107_{}_{:03d}.txt'.format(cpklName[:-7],cid))
			np.savetxt(outfilepath,res)
			t1 = time.time()
			print('frame {} in {}'.format(cpklName,t1 - t0))
		