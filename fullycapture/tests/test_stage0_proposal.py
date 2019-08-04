#test the rigid align method.
import glob
import os.path
from fullycapture.optimizer.fullyCapApp import fullyCapApp
import torch
import sys
import json
import numpy as np
import numpy.linalg as nlg
import cv2
import cPickle as pickle
import time
import fullycapture.data.dataIO as dataIO
import fullycapture.utils.config as cfg_utils
from fullycapture.utils.config import cfg


#Search.
#Instead of global align, use EDM for direct search.

def main():
    cfg_utils.merge_cfg_from_file('../../config/TotalCapture.yaml')
    cfg_utils.assert_and_infer_cfg()
    capUtils = fullyCapApp()
    capUtils.build_model_prior()
    seqname = '171204_pose6'

    joints_path = '/media/internal/domedb/{}/hdPose3d_total'.format(seqname)
    all_files = glob.glob(os.path.join(joints_path,'*.json'))
    all_files.sort()
    all_files_len = len(all_files)
    for idx,cfile in enumerate(all_files):
        frameId = dataIO.filepath_to_frameid(cfile)
        dds = dataIO.load_total_joints_3d(cfile)
        joints_total = []
        ids_total = []
        for dd in dds:
            j3d_t = np.zeros((65,4))
            j3d_t[0:25,:] = np.reshape(dd['joints25'],(-1,4))
            j3d_t[25:45,:] = np.reshape(dd['right_hand'],(-1,4))[1:,:]
            j3d_t[45:,:] = np.reshape(dd['left_hand'],(-1,4))[1:,:]
            joints_total.append(j3d_t)
            ids_total.append(dd['id'])
        joints_total = np.array(joints_total)
        theta,trans = capUtils.stage0_global_align_full(joints_total,gender='neutral')
        betas = np.zeros((theta.shape[0],10))
        print(ids_total)
        out_path = '/media/internal/domedb/{}/model_stage0_proposal/smpl_{:08d}.pkl'.format(seqname,frameId)
        with open(out_path,'wb') as fio:
            pickle.dump({'ids':ids_total,'beta':betas,'theta':theta,'trans':trans},fio)
        print('[{}/{}]'.format(idx+1,all_files_len))



if __name__=='__main__':
    export_prior_joints()