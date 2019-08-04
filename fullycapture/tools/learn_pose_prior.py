import cPickle as pickle
import os.path
import glob
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def kmeans_mocap_data():
	root_path = '/home/xiul/databag/neutrMosh'
	with open(os.path.join(root_path, 'file_list.pkl')) as fio:
		all_file = pickle.load(fio)
	n_file = len(all_file)

	batch_size = 1024*8
	n_k = 512
	kmeans = MiniBatchKMeans(
		n_clusters=n_k, random_state=0, batch_size=batch_size)
	print('total training data length {}'.format(n_file))

	file_ids = np.arange(0,n_file)
	file_ids = np.random.permutation(file_ids)
	c_file_id = 0

	max_iter = 650*10
	#650 batch,10 epoch

	data_rest = np.empty((0,72))
	
	for iter_num in range(max_iter):
		batch_data = np.empty((0,72))
		while batch_data.shape[0] < batch_size:
			#concate the rest into current batch
			batch_data = np.vstack((batch_data,data_rest))
			data_rest = np.empty((0,72))
			#load one file
			cfile = all_file[file_ids[c_file_id]]
			with open(cfile) as fio:
				dd = pickle.load(fio)
			
			if 'poses' in dd.keys():
				shard = dd['poses']
			elif 'new_poses' in dd.keys():
				shard = dd['new_poses']
			else:
				raise KeyError('pose key not found, given key is {}'.format(dd.keys()))
			shard_len = shard.shape[0]
			batch_len = batch_data.shape[0]

			rest_len = batch_size - batch_len
			if shard_len > rest_len:
				batch_data = np.vstack((batch_data,shard[:rest_len,:]))
				data_rest = shard[rest_len:,:]
			else:
				batch_data = np.vstack((batch_data,shard))
			

			c_file_id += 1
			if c_file_id == n_file:
				c_file_id = 0
				file_ids = np.random.permutation(file_ids)
		kmeans = kmeans.partial_fit(batch_data)
		print('kmeans running [{}/{}]'.format(iter_num+1,max_iter))
	print('kmeans clustering finished')

	np.savetxt('kmeans.txt', kmeans.cluster_centers_)


def mocap_data_to_joints():
	pose_prior = np.loadtxt('../../models/CMU_pose_kmeans_512.txt')
	#get joints from pose_prior. male,female and neutral.
	

	print(pose_prior.shape)
if __name__ == '__main__':
	#kmeans_mocap_data()
	mocap_data_to_joints()
