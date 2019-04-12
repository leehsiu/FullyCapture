#build SMPL model DCT prior based on the vertices positions.

import sys
import torch
import time
import numpy as np
import numpy.linalg as nlg
from totaldensify.model.batch_smpl import SmplModel
from totaldensify.utils.config import cfg
import totaldensify.utils.config as cfg_utils
import cPickle as pickle
import glob
import os.path
import json
import scipy.io
import totaldensify.geometry.geometry_process as geo_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_J_deg(J,J_parents):
    if J == -1:
        return 0 
    else:
        return 1+get_J_deg(J_parents[J],J_parents)
    
if __name__=='__main__':
    cfg_utils.merge_cfg_from_file('../../config/TotalCapture.yaml')
    cfg_utils.assert_and_infer_cfg()
    model_path = cfg.CAPTURE.MODEL_MALE
    smplwrapper = SmplModel(model_path)
    v_tmp = smplwrapper.v_template
    J_tmp = smplwrapper.J_template
    v_tmp_associate = np.ones((6890,1),dtype=np.int)*-1
    #for each vertices on v_tmp, calculate its distance to J_tmp and get the minimal one.
    for idx,vt in enumerate(v_tmp):
        J_dis = nlg.norm(vt - J_tmp,axis=1)
        ass = np.argmin(J_dis)
        v_tmp_associate[idx] = ass
    

    #how to get J_degree. get it to the root.
    J_deg = np.zeros((J_tmp.shape[0],1))
    J_par_tmp = smplwrapper.parents
    for idx in range(len(J_tmp)):
        J_deg[idx] = get_J_deg(idx,J_par_tmp)
    
    v_tmp_deg = np.zeros((6890,1))
    for idx,vt in enumerate(v_tmp_associate):
        v_tmp_deg[idx] = J_deg[vt]
    

    print(max(v_tmp_deg.flatten().tolist()))
    print(min(v_tmp_deg.flatten().tolist()))


    #np.savetxt('../../models/SMPL_vts_degree.txt',v_tmp_deg)


    #
    sat_start = [0]+np.linspace(1,40,12)

    dct_weight = np.zeros((6890,100))
    #generate dct_weight based on degree
    print(len(sat_start))
    #first try. linder scaling from 0->n

    # deg_sat = [10,50]
    for idx,vt_d in enumerate(v_tmp_deg):
        vt_d_int = int(vt_d)

        num = 100 - int(sat_start[vt_d_int])
        if(vt_d_int<=2):
            dct_weight[idx,:] = np.linspace(0,1,100)*5.0
            dct_weight[idx,40:] = 1e2
        
        else:
            dct_weight[idx,int(sat_start[vt_d_int]):] = np.linspace(0,1,num)*1.5
    
    np.savetxt('../../models/dct_weight.txt',dct_weight)
    # for i in range(0,6890,20):
    #     plt.plot(dct_weight[i])
    
    # plt.show()
