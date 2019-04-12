import totaldensify.data.dp_methods as dp_utils
import os
import os.path
import glob
import cv2
import numpy as np
import time
if __name__=='__main__':
	
	rootpath = '/home/xiul/databag/dome_sptm/171204_pose6' 
	outpath = '/home/xiul/databag/dome_sptm/171204_pose6/dp_vts_xiu'
	if not os.path.isdir(outpath):
		os.mkdir(outpath)

	pklFiles = glob.glob(os.path.join(rootpath,'dp_xiu','*_IUV.png'))
	pklFiles.sort()
	for cpkl in pklFiles:
		cpklName = os.path.basename(cpkl)
		cpklName = os.path.splitext(cpklName)[0]

		t0 = time.time()
		uv_img = cv2.imread(cpkl)

		res =dp_utils.dp_uvi_to_verts(uv_img,dp_utils.dp_colors_smpl)
		outfilepath = os.path.join(outpath,'{}_{:03d}.txt'.format(cpklName[:-4],0))
		np.savetxt(outfilepath,res)
		t1 = time.time()
		print('frame {} in {}'.format(cpklName,t1 - t0))