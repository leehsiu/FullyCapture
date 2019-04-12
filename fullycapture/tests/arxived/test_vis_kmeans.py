import sys
import torch
import numpy as np
import time
import totaldensify.optimizer.torch_smplify as smplify_util
from totaldensify.model.batch_adam_torch import AdamModelTorch
from totaldensify.model.batch_smpl_torch import SmplModelTorch
import totaldensify.data.dataIO as dataIO
from totaldensify.vis.glut_viewer import glut_viewer
import cPickle as pickle
import glob
import os.path

def test_coco25_smpl():
    # adam_path = '/home/xiul/workspace/TotalDensify/models/adamModel_with_coco25_reg.pkl'
    # totalModel = AdamModelTorch(pkl_path=adam_path, reg_type='coco25')
    smpl_path = '/home/xiul/workspace/TotalDensify/models/smplModel_with_coco25_reg.pkl'
    smplModel = SmplModelTorch(pkl_path=smpl_path,reg_type='coco25')

    poses = np.loadtxt('../data/kmeans.txt')

    #frame_id = 80
    #model_pkls = [model_pkls[65]]
    print(poses.shape)
    n_batch = poses.shape[0]
    #model_pkls = model_pkls[:100]
    # n_batch = len(model_pkls)
    betas = np.zeros((n_batch, 10))
    thetas = poses.reshape(n_batch,24,3)
    trans = np.zeros((n_batch, 1, 3))
    thetas[:,:3,:] = 0

    # vts_total=  []
    # vts_weight_total = []

    # for i_file, model_file in enumerate(model_pkls):
    #     base_name = os.path.basename(model_file)
    #     base_name = os.path.splitext(base_name)[0]
    #     frame_id = int(base_name.split('_')[1])
    #     with open(model_file) as fio:
    #         cdd = pickle.load(fio)
    #     betas[i_file] = cdd[0]['betas']
    #     thetas[i_file] = np.reshape(cdd[0]['pose'], (-1, 3))
    #     trans[i_file] = cdd[0]['trans']
    #     vts_3d,vts_weight = dataIO.load_dp_vts_3d('/home/xiul/databag/dome_sptm/171204_pose6/dp_pcd_naive',frame_id,1)
    #     vts_total.append(vts_3d)
    #     vts_weight_total.append(vts_weight[:,None])

    
    # vts_total = np.array(vts_total).astype(np.float32)
    # vts_weight_total = np.array(vts_weight_total).astype(np.float32)

    
    betas_cu = torch.tensor(betas,dtype=torch.float32).cuda()
    thetas_cu = torch.tensor(thetas,dtype=torch.float32).cuda()
    trans_cu = torch.tensor(trans,dtype=torch.float32).cuda()

    # #init ground truth data
    v3d_gt, j3d_gt = smplModel(betas_cu, thetas_cu, reg_type='coco25')
    v3d_gt = v3d_gt + trans_cu
    j3d_gt = j3d_gt + trans_cu
    


    calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'


    glRender = glut_viewer(1280, 720)
    glRender.loadCalibs(calibFile)

    v3d_pred_cpu = v3d_gt.detach().cpu().numpy()
    j3d_pred_cpu = j3d_gt.detach().cpu().numpy()

    glRender.load_data(v3d_pred_cpu, j3d_pred_cpu)
    glRender.load_gt_data(v3d_pred_cpu,j3d_pred_cpu)
    glRender.init_GLUT(sys.argv)

if __name__=='__main__':
    test_coco25_smpl()
