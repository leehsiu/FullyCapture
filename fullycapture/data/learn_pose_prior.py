import cPickle as pickle
import os.path
import glob
import numpy as np
from sklearn.cluster import MiniBatchKMeans
if __name__=='__main__':
    root_path = '/home/xiul/databag/neutrMosh'
    # all_files = []
    # all_record = []

    # subdirs = glob.glob(os.path.join(root_path,'*'))
    # for subdir in subdirs:
    #     subseq = glob.glob(os.path.join(subdir,'*'))
    #     for seq in subseq:
    #         cfiles = glob.glob(os.path.join(seq,'*.pkl'))
    #         all_files.extend(cfiles)

    # print('all Mosh data found {}'.format(len(all_files)))
    with open(os.path.join(root_path,'file_list.pkl')) as fio:
        all_file = pickle.load(fio)
    #print(len(all_file))
    n_file = len(all_file)
    batch_size = 4096
    n_k = 512
    kmeans = MiniBatchKMeans(n_clusters=n_k,random_state=0,batch_size=batch_size)


    c_data = np.empty((0,72))
    c_rest = np.empty((0,72))
    iter_num = 0
    c_file_id = 0

    max_epoch = 5

    for epoch in range(max_epoch):
        c_file_id = 0
        while c_file_id<n_file:
            while c_data.shape[0]<batch_size:
                c_data = np.vstack((c_data,c_rest))

                cfile = all_file[c_file_id]
                c_file_id +=1
                with open(cfile) as fio:
                    dd = pickle.load(fio)
                if 'poses' in dd.keys():
                    c_pose = dd['poses']
                elif 'new_poses' in dd.keys():
                    c_pose = dd['new_poses']
                else:
                    print(dd.keys())
                    raise TypeError('dd')
                c_data_len = c_data.shape[0]
                c_num = c_pose.shape[0]
                c_resi_len = 1024 - c_data_len
                if c_num > c_resi_len:
                    c_data = np.vstack((c_data,c_pose[:c_resi_len,:]))
                    c_rest = c_pose[c_resi_len:,:]
                else:
                    c_data = np.vstack((c_data,c_pose[:c_resi_len,:]))
                    c_rest = np.empty((0,72))
            iter_num += 1
            kmeans = kmeans.partial_fit(c_data)
            c_data = np.empty((0,72))
            print('Epoch {}:iter {} file loaded[{}/{}]'.format(epoch,iter_num,c_file_id,n_file))
            #print(kmeans.cluster_centers_)

    np.savetxt('kmeans.txt',kmeans.cluster_centers_)
