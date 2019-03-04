import glob
import totaldensify.data.dataIO as dataIO
import totaldensify.geometry.triangulate as mvg_utils
import numpy as np
import time
import cPickle as pickle
import os.path
def DP_Triangulation_withSMPL(pts2d, P_matrices):
	"""
	pts2d as VxNx2  V views, N points, 2(x,y)
	P_matrices as Vx3x4
	SMPL_Prior as N*3
	"""
	N = pts2d.shape[1]
	V = pts2d.shape[0]
	t0 = time.time()
	pts_tracks = list(map(lambda x: np.where(pts2d[:, x, 0] > 0), range(N)))
	pts3d = np.zeros((N, 4))
	#pts3d as Nx4, (pid,x,y,z)
	pts3d_err = np.zeros(N) - 1
	pts3d[:, 0] = -1
	ptRange = range(N)
	#print pts_tracks
	for i in ptRange[::3]:
		if(pts_tracks[i][0].shape[0] >= 4):
			#p3d,pts3d_err[i] = DLTtriangulate.triangulate_onepoint_naive(pts2d[pts_tracks[i][0],i,:-3:-1],P_matrices[pts_tracks[i][0],:,:])
			p3d, pts3d_err[i] = mvg_utils.triangulate_onepoint_naive(
				pts2d[pts_tracks[i][0], i, :-3:-1], P_matrices[pts_tracks[i][0], :, :])
			if p3d is not None:
				pts3d[i, 1:4] = p3d
				pts3d[i, 0] = i
		#print i
		else:
			pts3d[i, 0] = -1
	return pts3d, pts3d_err
if __name__ == '__main__':
	#seqName = '170221_haggling_b1'
	seqName = '171204_pose6'
	datapath = '/home/xiul/databag/dome_sptm/{}/dp_vts_e2e'.format(seqName)
	all_c_params = glob.glob(
		'/home/xiul/databag/dome_sptm/{}/gt_pkl/*.pkl'.format(seqName))
	all_c_params.sort()
	viewnodes = range(31)
	Pmat = dataIO.parse_dome_calibs(
		'/home/xiul/databag/dome_sptm/{}/calibration_{}.json'.format(seqName, seqName), viewnodes)

	for cpkl in all_c_params:
		with open(cpkl) as fio:
			c_param = pickle.load(fio)
		cpklName = os.path.basename(cpkl)
		cpklName = os.path.splitext(cpklName)[0]
		frameId = int(cpklName.split('_')[1])
		perId = 1
		pts_2d = dataIO.load_dp_vts_2d(datapath, frameId,perId)
		pts3d, pts3derr = DP_Triangulation_withSMPL(pts_2d, Pmat)
		print('#Point Cloud generated:{} to '.format(frameId))
		import numpy as np
		np.savetxt('/home/xiul/databag/dome_sptm/{}/dp_pcd_naive/pcd_{:08d}_{:03d}.txt'.format(
			seqName, frameId, perId), pts3d)
		np.savetxt('/home/xiul/databag/dome_sptm/{}/dp_pcd_naive/pcd_{:08d}_{:03d}_err.txt'.format(
			seqName, frameId, perId), pts3derr)
