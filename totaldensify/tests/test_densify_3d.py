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
    adam_path = '/home/xiul/workspace/TotalDensify/models/adamModel_with_coco25_reg.pkl'
    totalModel = AdamModelTorch(pkl_path=adam_path, reg_type='coco25')
    smpl_path = '/home/xiul/workspace/TotalDensify/models/smplModel_with_coco25_reg.pkl'
    smplModel = SmplModelTorch(pkl_path=smpl_path,reg_type='coco25')


    model_path = '/home/xiul/databag/dome_sptm/171204_pose6/gt_pkl'

    model_pkls = glob.glob(os.path.join(model_path, '*.pkl'))
    model_pkls.sort()

    #frame_id = 80
    #model_pkls = [model_pkls[65]]


    #model_pkls = model_pkls[:100]
    n_batch = len(model_pkls)
    betas = np.zeros((n_batch, 30))
    thetas = np.zeros((n_batch, 62, 3))
    trans = np.zeros((n_batch, 1, 3))
    joint_mask = np.zeros((n_batch,25,3))

    vts_total=  []
    vts_weight_total = []

    for i_file, model_file in enumerate(model_pkls):
        base_name = os.path.basename(model_file)
        base_name = os.path.splitext(base_name)[0]
        frame_id = int(base_name.split('_')[1])
        with open(model_file) as fio:
            cdd = pickle.load(fio)
        betas[i_file] = cdd[0]['betas']
        thetas[i_file] = np.reshape(cdd[0]['pose'], (-1, 3))
        trans[i_file] = cdd[0]['trans']
        vts_3d,vts_weight = dataIO.load_dp_vts_3d('/home/xiul/databag/dome_sptm/171204_pose6/dp_pcd_naive',frame_id,1)
        vts_total.append(vts_3d)
        vts_weight_total.append(vts_weight[:,None])

    
    vts_total = np.array(vts_total).astype(np.float32)
    vts_weight_total = np.array(vts_weight_total).astype(np.float32)

    betas_cu = torch.tensor(betas,dtype=torch.float32).cuda()
    thetas_cu = torch.tensor(thetas,dtype=torch.float32).cuda()
    trans_cu = torch.tensor(trans,dtype=torch.float32).cuda()

    #init ground truth data
    v3d_gt, j3d_gt = totalModel(betas_cu, thetas_cu, reg_type='coco25')
    v3d_gt = v3d_gt + trans_cu
    j3d_gt = j3d_gt + trans_cu
    

    recon = smplify_util.BodyRecon()

    j3d_t = j3d_gt.detach().cpu().numpy()
    v3d_t = v3d_gt.detach().cpu().numpy()

    # vtd_3d = vts_3d[None,:,:]
    # vts_weight = vts_weight[None,:,None]
    #vts_zeros_weight = np.zeros((1,6890,1))

    n_iter = 100
    t0 = time.time()

    #Init param
    betas_zero = np.zeros((1, 10)).astype(np.float32)
    thetas_zero = np.zeros((n_batch, 24, 3)).astype(np.float32)
    trans_zero = np.zeros((n_batch, 1, 3)).astype(np.float32)
    joint_mask = np.zeros((n_batch,25,3))
    joint_mask[:,0:18,:] = 1.0
    init_param = {'betas': betas_zero, 'thetas': thetas_zero,'trans':trans_zero}

    #betas_n, thetas_n,trans_n= recon.smplify3d_adam(
    #    smplModel, j3d_t, init_param,joint_mask=joint_mask,n_iter=n_iter, reg_type='coco25')
    betas_n, thetas_n,trans_n= recon.densify3d_adam(
       smplModel, j3d_t,vts_total ,init_param,joint_mask=joint_mask,verts_mask=vts_weight_total,n_iter=n_iter, reg_type='coco25')
    
    t1 = time.time()

    print('Time for each model {} sec'.format((t1 - t0)/(n_batch)))

    betas_cu_n = torch.tensor(betas_n).cuda()
    thetas_cu_n = torch.tensor(thetas_n).cuda()
    trans_cu_n = torch.tensor(trans_n).cuda()

    v3d_n, j3d_n = smplModel(betas_cu_n, thetas_cu_n, reg_type='coco25')
    v3d_n = v3d_n + trans_cu_n
    j3d_n = j3d_n + trans_cu_n

    calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'


    glRender = glut_viewer(1280, 720)
    glRender.loadCalibs(calibFile)

    v3d_pred_cpu = v3d_n.detach().cpu().numpy()
    j3d_pred_cpu = j3d_n.detach().cpu().numpy()

    glRender.load_data(v3d_pred_cpu, j3d_pred_cpu)
    glRender.load_gt_data(vts_total,j3d_t)
    glRender.init_GLUT(sys.argv)

if __name__=='__main__':
    test_coco25_smpl()
