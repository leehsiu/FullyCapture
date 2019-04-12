import datetime
from skimage import measure
import scipy.sparse
import os
import torch
import torch.nn.functional as torchF
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.EGL import *
import numpy as np
import numpy.linalg as nlg
import cPickle as pickle
import json
import argparse
import time
import glob
import os.path
import copy
import sklearn.preprocessing
import cv2
import scipy.io
import sys
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from totaldensify.vis.egl_render import EglRender
from totaldensify.model.batch_smpl import SmplModel
import totaldensify.vis.plot_vis as plot_vis
import totaldensify.geometry.geometry_process as geo_utils
import matplotlib.pyplot as plt

#DensePose and Adam Global Items 
smpl_iuv_path = '../../models/smpl_iuv.txt'

smpl_iuv = np.loadtxt(smpl_iuv_path)
smpl_iuv[:,0] = smpl_iuv[:,0]/255.0
smpl_iuv = smpl_iuv[:,::-1]
print(smpl_iuv)



def main_func():  
    # Init EGL 

    seqPath = '/home/xiul/databag/dbfusion/record0'
    allImgs = glob.glob(os.path.join(seqPath,'fullcolor/*.png'))
    allImgs.sort()

    allPoses = np.loadtxt(os.path.join(seqPath,'pose_parameters_per_frame.txt'))
    allBetas =np.loadtxt(os.path.join(seqPath,'shape_parameters.txt'))
    camK = np.loadtxt(os.path.join(seqPath,'cam_params.txt'),delimiter=',')

    smplWrapper = SmplModel(pkl_path='../../models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl',reg_type='legacy')
    
    allTrans = allPoses[:,3:6]
    allPose = np.hstack((allPoses[:,0:3],allPoses[:,6:]))
    allBetas = np.tile(allBetas,(allPoses.shape[0],1))
    v_all,_ = smplWrapper(allBetas,allPose)
    v_all += np.expand_dims(allTrans,axis=1)

    f_smpl = smplWrapper.f
    n_frame = len(allTrans)

    egl = EglRender(1920,1080)
    egl.set_flat(True)
    R = np.eye(3)
    t = np.zeros((1,3))
    #egl.set_shade_model(GL_FLAT)
    for i in range(n_frame):
        vt = v_all[i]
        vn = geo_utils.vertices_normals(f_smpl,vt)
        #egl.enable_light(True)
        
        vis_image0,_ = egl.render_obj(vt,smpl_iuv,vn,f_smpl,R,t.T,camK)
        img_path = allImgs[i]
        img = cv2.imread(img_path)
        #img0 = plot_vis.make_overlay(vis_image0[:,:,:3],img,vis_image0[:,:,3])
        img_base = os.path.basename(img_path)
        img_base = os.path.splitext(img_base)[0]
        out_path = '/home/xiul/databag/dbfusion/record0/dp_gt_flat'
        out_file_name = os.path.join(out_path,'{}_IUV.png'.format(img_base))
        print(img_base)

        cv2.imwrite(out_file_name,vis_image0)
        #cv2.imshow('img0',vis_image0)
        #cv2.waitKey(1)
        
		# img1 = plot_vis.make_overlay(vis_image1[:,:,:3],img,vis_image1[:,:,3])

if __name__ == '__main__':
    main_func()