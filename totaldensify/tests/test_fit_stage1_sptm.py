import sys
import torch
import numpy as np
import time
import totaldensify.optimizer.std_capture as capture_utils
from totaldensify.model.batch_smpl_torch import SmplModelTorch
import totaldensify.data.dataIO as dataIO
from totaldensify.vis.glut_viewer import glut_viewer
from totaldensify.utils.config import cfg
import totaldensify.utils.config as cfg_utils
import cPickle as pickle
import glob
import os.path

if __name__=='__main__':

    cfg_utils.merge_cfg_from_file('../../config/TotalCapture.yaml')
    cfg_utils.assert_and_infer_cfg()

    model_path = cfg.CAPTURE.MODEL
    capUtils = capture_utils.StdCapture(model_path,reg_type='total')

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

    #frame_id = 80
    #model_pkls = [model_pkls[65]]
    n_batch = joints_total.shape[0]

    vts_total=  []
    vts_weight = []

    
    for frame_id in range(500,600):
        v_3d,v_weight = dataIO.load_dp_vts_3d('/home/xiul/databag/dome_sptm/171204_pose6/dp_pcd_naive',frame_id,1)
        vts_total.append(v_3d)
        vts_weight.append(v_weight[:,None])

    vts_total = np.array(vts_total).astype(np.float32)
    vts_weight = np.array(vts_weight).astype(np.float32)
    #vts_weight_total[:,:,:] = 0

    with open('./stage0.pkl') as fio:
        dd = pickle.load(fio)

    init_param = {'betas':dd['betas'][0],'thetas':dd['pose'],'trans':dd['trans']}

    betas_n,thetas_n,trans_n = capUtils.fit_stage1_spatial_temporal(joints_total[:,:,:3],joints_total[:,:,3],vts_total,vts_weight,init_param,400)

    with open('stage1.pkl','wb') as fio:
        pickle.dump({'betas':betas_n,'pose':thetas_n,'trans':trans_n},fio)


    v_all,J_all = capUtils.smplWrapperCPU(betas_n,thetas_n)
    v_all += trans_n
    J_all += trans_n

    

    calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'


    glRender = glut_viewer(1280, 720)
    glRender.loadCalibs(calibFile)


    glRender.load_data(v_all, J_all[:,:25,:])
    glRender.load_gt_data(vts_total,joints_total[:,:25,:][:,:,:3])
    glRender.init_GLUT(sys.argv)