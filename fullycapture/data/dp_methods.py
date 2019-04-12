import numpy as np
import cv2
import cPickle as pickle
import time
import scipy.spatial
import scipy.io
import os
import glob

smpldp_Path = '/home/xiul/workspace/TotalDensify/models/smpl_iuv.txt'
smpldp = np.loadtxt(smpldp_Path)

dp_colors_smpl = smpldp 


def dp_uvi_to_verts(im_iuv,table_iuv,ignore_part=[3,4,23,24]):
	nVerts = table_iuv.shape[0]
	nParts = 24
	iChan = im_iuv[:,:,0].flatten()
	uChan = im_iuv[:,:,1].flatten()/256.0
	vChan = im_iuv[:,:,2].flatten()/256.0
	uvChan = np.vstack((uChan,vChan)).T

	posChan = np.arange(0,len(iChan))

	iTable = table_iuv[:,0].flatten().astype(np.int)
	uvTable = table_iuv[:,1:3]

	res = np.zeros((nVerts,3),np.float) #as, err,x,y
	for partId in range(1,nParts+1):
		#no hand and face
		if partId in ignore_part:
			continue
		tableMask = iTable==partId
		tableUV = uvTable[tableMask,:]
		
		imgMask = iChan==partId
		imgUV = uvChan[imgMask,:]

		imgIndices = posChan[imgMask]
		
		if len(imgUV) < 1:
			continue

		dist = scipy.spatial.distance.cdist(tableUV,imgUV,'cityblock')

		nn_pos = np.argmin(dist,axis=1)
		nn_dis = np.min(dist,axis=1)

		res[tableMask,0] = nn_dis
		# print posChan[imgIndices[nn_pos]] % im_iuv.shape[1] 
		# print posChan[imgIndices[nn_pos]] // im_iuv.shape[1]
		res[tableMask,1] = posChan[imgIndices[nn_pos]] % im_iuv.shape[1] #x, col
		res[tableMask,2] = posChan[imgIndices[nn_pos]] // im_iuv.shape[1] #y, row

	return res

if __name__=='__main__':

	# seqname = '171204_pose6'
	# rootpath = '/home/xiul/databag/dome_sptm/{}/dp_e2e'.format(seqname)  #where the iuv_images/undistorted
	# outpath = '/home/xiul/databag/dome_sptm/{}/dp_vts_e2e'.format(seqname)     #where to store the features
	# if not os.path.isdir(outpath):
	# 	os.mkdir(outpath)

	# pklFiles = glob.glob('/home/xiul/databag/dome_sptm/171204_pose3/gt_pkl/*.pkl')
	# pklFiles.sort()

	# for viewId in range(0,31):
	# 	for cpkl in pklFiles:
	# 		cpklName = os.path.basename(cpkl)
	# 		cpklName = os.path.splitext(cpklName)[0]
	# 		frameId = int(cpklName.split('_')[1])
	# 		uv_name = os.path.join(rootpath,'00_{:02d}_{:08d}_IUV.png'.format(viewId,frameId))
	# 		inds_name = os.path.join(rootpath,'00_{:02d}_{:08d}_INDS_GT.png'.format(viewId,frameId)) #INDS_GT means after associate
	# 		#inds_name = os.path.join(rootpath,'00_{:02d}_{:08d}_INDS.png'.format(viewId,frameId)) #INDS_GT means after associate
	# 		t0 = time.time()

	# 		if os.path.isfile(uv_name):
	# 			uv_img = cv2.imread(uv_name)
	# 			if os.path.isfile(inds_name):
	# 				inds_img = cv2.imread(inds_name)
	# 				inds_set = list(set(inds_img[:,:,0].flatten().tolist()))
	# 				inds_set = inds_set[1:] #each person
	# 			else: #Single person case
	# 				inds_img = np.zeros(uv_img.shape,uv_img.dtype)
	# 				inds_img[uv_img[:,:,0]>0,:] = 1
	# 				inds_set = [1]

	# 			for cid in inds_set:
	# 				c_uv_img = uv_img.copy()
	# 				c_uv_img[inds_img[:,:,0]!=cid,:] = 0
	# 				res =dp_uvi_to_verts(c_uv_img,dp_colors_smpl)
	# 				outfilepath = os.path.join(outpath,'00_{:02d}_{:08d}_{:03d}.txt'.format(viewId,frameId,cid))
	# 				np.savetxt(outfilepath,res)
	# 		t1 = time.time()
	# 		print('view {}: frame {} in {}'.format(viewId,frameId,t1 - t0))
	seqname = 'dslr_dance'
	rootpath = '/home/xiul/databag/dslr_dance/densepose'  #where the iuv_images/undistorted
	outpath = '/home/xiul/databag/dslr_dance/dense_feature'     #where to store the features
	if not os.path.isdir(outpath):
		os.mkdir(outpath)

	pklFiles = glob.glob('/home/xiul/databag/dslr_dance/images/*.png')
	pklFiles.sort()
	for cpkl in pklFiles:
		cpklName = os.path.basename(cpkl)
		cpklName = os.path.splitext(cpklName)[0]
		#frameId = int(cpklName.split('_')[1])
		uv_name = os.path.join(rootpath,'{}_IUV.png'.format(cpklName))
		#inds_name = os.path.join(rootpath,'00_{:02d}_{:08d}_INDS_GT.png'.format(viewId,frameId)) #INDS_GT means after associate
		#inds_name = os.path.join(rootpath,'00_{:02d}_{:08d}_INDS.png'.format(viewId,frameId)) #INDS_GT means after associate
		t0 = time.time()
		if os.path.isfile(uv_name):
			uv_img = cv2.imread(uv_name)
			#inds_set = [1]
			res =dp_uvi_to_verts(uv_img,dp_colors_smpl)
			outfilepath = os.path.join(outpath,'{}.txt'.format(cpklName))
			np.savetxt(outfilepath,res)
		t1 = time.time()
		print('frame {} in {}'.format(cpklName,t1 - t0))