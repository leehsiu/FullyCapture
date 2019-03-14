#test the rigid align method.
import glob
import os.path
from totaldensify.optimizer.std_capture import StdCapture
import torch
import sys
import json
import numpy as np
import numpy.linalg as nlg
import cv2
import cPickle as pickle
from totaldensify.vis.glut_viewer import glut_viewer
import time
import totaldensify.data.dataIO as dataIO
import totaldensify.utils.config as cfg_utils
from totaldensify.utils.config import cfg

if __name__=='__main__':

    cfg_utils.merge_cfg_from_file('../../config/TotalCapture.yaml')
    cfg_utils.assert_and_infer_cfg()

    model_path = cfg.CAPTURE.MODEL
    capUtils = StdCapture(model_path,reg_type='total')

    joints_path = '/media/internal/domedb/171204_pose6/hdPose3d_total'
    all_files = glob.glob(os.path.join(joints_path,'*.json'))
    all_files.sort()

    #start_id, 136
    #our range id, 500-599

    frame_range = range(500-136,500-136+100)
    joints_total = []
    for frame_id in frame_range:
        dd = dataIO.load_total_joints_3d(all_files[frame_id])[0]
        j3d_t = np.zeros((65,4))
        j3d_t[0:25,:] = np.reshape(dd['joints25'],(-1,4))

        j3d_t[25:45,:] = np.reshape(dd['right_hand'],(-1,4))[1:,:]
        j3d_t[45:,:] = np.reshape(dd['left_hand'],(-1,4))[1:,:]
        joints_total.append(j3d_t)
    
    joints_total = np.array(joints_total)

    t0 = time.time()
    theta,trans = capUtils.stage0_global_align_full(joints_total,reg_type='total')
    t1 = time.time()




    print('stage0 direct global align time :{}'.format(t1-t0))
    n_batch = theta.shape[0]
    betas = np.zeros((n_batch,10))

    for i in range(n_batch):
        betas_n,theta_n,trans_n = capUtils.cops.smpl_fit_stage1(joints_total[i],betas[i],theta[i]*-1,trans[i],0,True,False)

        trans[i] = trans_n
        theta[i] = theta_n.reshape(-1,3)*-1
        betas[i] = betas_n
    t2 = time.time()

    print('stage0 LM C++ refine time {}'.format(t2-t1))

    with open('stage0.pkl','wb') as fio:
        pickle.dump({'betas':betas,'pose':theta,'trans':trans},fio)
    
    
    v_all,J_all = capUtils.smplWrapperCPU(betas,theta)
    v_all += trans
    J_all += trans

    

    calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'


    glRender = glut_viewer(1280, 720)
    glRender.loadCalibs(calibFile)


    glRender.load_data(v_all, J_all[:,:25,:])
    glRender.load_gt_data(v_all,joints_total[:,:25,:][:,:,:3])
    glRender.init_GLUT(sys.argv)