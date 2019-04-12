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
from totaldensify.model.batch_adam import TotalModel
from totaldensify.model.batch_adam_torch import AdamModelTorch

import totaldensify.geometry.geometry_process as geo_utils
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,2)
#DensePose and Adam Global Items 
adamPath = '/media/posefs3b/Users/xiu/adams/adamModel_with_coco25_reg.pkl'
adamWrapper = TotalModel(pkl_path=adamPath,reg_type='coco25')
adamCuda = AdamModelTorch(pkl_path=adamPath,reg_type='coco25')

adamdpPath = '/media/posefs3b/Users/xiu/adams/adam_to_uvi.pkl'
with open(adamdpPath) as fio:
    adamdpPkl = pickle.load(fio)


dp_mask_id = np.array([0,1,1,2,3,4,5,6,7,6,7,8,9,8,9,10,11,10,11,12,13,12,13,14,14])

dp_colors_adam = np.array(adamdpPkl['adam_to_uvi'])

dp_adam_I = dp_colors_adam[:,2].astype(np.int)
dp_adam_mask = dp_mask_id[dp_adam_I]




# Egl Rendering Global Items

total_calibs = []
seq_hash = {}

test_set= ['170221_haggling_b1',
          '170221_haggling_b2',
          '170221_haggling_b3',
          '170228_haggling_b1',
          '170228_haggling_b2',
          '170228_haggling_b3',
          '171204_pose5',
          '171204_pose6']

def pre_load_calibs():
    global total_calibs,seq_hash
    with open('/media/posefs3b/Users/xiu/adams/total_calibs.json') as fio:
        total_calibs = json.load(fio)
    all_seq_name = total_calibs.keys()
    seq_hash = {}
    for idx,seqname in enumerate(all_seq_name):
        seq_hash['{}'.format(seqname)] = idx




def from_reprojection(np_uvi_masked,np_binary_mask,vts_2d,vis_2d):
    cAnno = None
    fortran_binary_mask = np.asfortranarray(np_binary_mask)
    coco_mask = mask_util.encode(fortran_binary_mask)
    coco_area = mask_util.area(coco_mask)
    coco_bbox = mask_util.toBbox(coco_mask)
    coco_bbox = coco_bbox.astype(np.int)
    bbox_th = 35
    if int(coco_bbox[2])<=bbox_th and int(coco_bbox[3])<=bbox_th:
        return cAnno

    coco_segments = polygon_from_binary_mask(np_binary_mask,tolerance=3)
    np_uvi_crop = np_uvi_masked[coco_bbox[1]:coco_bbox[1]+coco_bbox[3],:][:,coco_bbox[0]:coco_bbox[0]+coco_bbox[2]]

    bbox_area = coco_bbox[2]*coco_bbox[3]

    #sample_num = int(round(np_area_256/(sample_patch*sample_patch)))
    if float(coco_area)/bbox_area<0.2:
        return cAnno
    

    np_mask_256 = cv2.resize(np_uvi_crop,(256,256),interpolation=cv2.INTER_NEAREST)

    coco_dp_masks_rle = []
    for c_dp_idx in range(1,15):
        c_idx_mask = np.zeros((256,256),dtype=np.uint8)
        c_idx_mask[np_mask_256[:,:,0]==c_dp_idx] = c_dp_idx
        c_idx_mask_fortran = np.asfortranarray(c_idx_mask)
        c_rle_mask = mask_util.encode(c_idx_mask_fortran)
        coco_dp_masks_rle.append(c_rle_mask)

    #get the visibility.
    coco_dp_I = []
    coco_dp_U = []
    coco_dp_V = []
    coco_dp_X = []
    coco_dp_Y = []

    import random    
    for ii in range(1,15):
        vis_ids, = np.where(np.logical_and(vis_2d<1e-1,dp_adam_mask==ii))
        #print(vis_ids)
        if(len(vis_ids)<14):
            sampled_id = vis_ids
        else:
            sampled_id = random.sample(vis_ids,14)
        for cid in sampled_id:
            coco_dp_I.append(dp_colors_adam[cid,0])
            coco_dp_U.append(dp_colors_adam[cid,1])
            coco_dp_V.append(dp_colors_adam[cid,2])
            c_x = vts_2d[cid,0]*640+640
            c_y = vts_2d[cid,1]*360+360
            c_x_256 = (c_x - coco_bbox[0])/coco_bbox[2]*256.0
            c_y_256 = (c_y - coco_bbox[1])/coco_bbox[3]*256.0
            coco_dp_X.append(c_x_256)
            coco_dp_Y.append(c_y_256)    
    
    cAnno = {"segmentation":coco_segments,"dp_masks":coco_dp_masks_rle,"area":int(coco_area),"bbox":coco_bbox.tolist(),
            "dp_I":coco_dp_I,"dp_U":coco_dp_U,"dp_V":coco_dp_V,"dp_x":coco_dp_X,"dp_y":coco_dp_Y}
    return cAnno
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def polygon_from_binary_mask(np_binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated,for simplificity
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    contours = measure.find_contours(np_binary_mask, 0.5)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

def create_image_info(seqname,camid,frameid):
    file_name = '{}_{:02d}_{:06d}.jpg'.format(seqname,camid,frameid)
    image_id = '{:02d}{:02d}{:06d}'.format(seq_hash[seqname],camid,frameid)
    license_id=1
    coco_url=""
    flickr_url=""
    date_captured=datetime.datetime.utcnow().isoformat(' ')
    image_info = {
            "id": int(image_id),
            "file_name": file_name,
            "width": 1280,
            "height": 720,
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info


def is_full_green(img):
    greenColor = [0,135,0]

    if (img[3,3]==greenColor).all() and (img[3,-3] == greenColor).all()\
        and (img[-3,-3] == greenColor).all() and (img[-3,3] ==greenColor).all():
        return True
    else:
        return False



def paramfile_to_render_objects(param_file):

    vc_all = []
    f_all = []
    vis_inds = []

    with open(param_file) as f:
        SMPLParams = pickle.load(f)
    trans = []
    betas = []
    poses = []
    ids = []
    for idx, cParam in enumerate(SMPLParams):
        trans.append(cParam['trans'])
        betas.append(cParam['betas'])
        poses.append(cParam['pose'])
        ids.append(cParam['id'])
        f_all.append(adamWrapper.f+idx*adamWrapper.size[0])
        vc = np.zeros((18540,3),dtype=np.float32)
        vc[:,0] = dp_adam_mask
        vc[:,1] = cParam['id']+1
        vc_all.append(vc)

    f_all = np.stack(f_all,axis=0)
    vc_all = np.stack(vc_all,axis=0)

    f_all = np.reshape(f_all,(-1,3))
    trans = np.asarray(trans,np.float32)
    betas = np.asarray(betas,np.float32)
    poses = np.asarray(poses,np.float32)


    betas_cuda = torch.tensor(betas,dtype=torch.float32).cuda()
    thetas_cuda = torch.tensor(poses,dtype=torch.float32).cuda()
    trans_cuda = torch.tensor(trans[:,None,:],dtype=torch.float32).cuda()


    v_cuda,_ = adamCuda(betas_cuda,thetas_cuda)
    v_cuda += trans_cuda


    vt_all = v_cuda.view(1,-1,3)

    return vt_all,vc_all,f_all,ids



def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--shard_id',
        dest='shard_id',
        default=None,
        type=str
    )
    parser.add_argument(
        '--skip',
        dest='skip',
        default=30,
        type=int
    )
    parser.add_argument(
        '--segid',
        dest='segid',
        default=0,
        type=int
    )
    parser.add_argument(
        '--mode',
        dest='mode',
        default='coco',
        type=str
    )
    args = parser.parse_args()
    return args


def uvi_to_label(np_uvi_masked,np_binary_mask):
    fortran_binary_mask = np.asfortranarray(np_binary_mask)
    coco_mask = mask_util.encode(fortran_binary_mask)
    coco_bbox = mask_util.toBbox(coco_mask)
    coco_bbox = coco_bbox.astype(np.int)

    bbox_th = 35
    if int(coco_bbox[2])<=bbox_th and int(coco_bbox[3])<=bbox_th:
        return None , None
    np_uvi_crop = np_uvi_masked[coco_bbox[1]:coco_bbox[1]+coco_bbox[3],:][:,coco_bbox[0]:coco_bbox[0]+coco_bbox[2]]
    return coco_bbox,np_uvi_crop
    



def main_regen():  
    # Init EGL 

    annFile = '/home/xiul/databag/COCO/annotations/dome_dense_2018_train_strip.json'

    mainRender = EglRender(1280,720)
    mainRender.enable_light(False)
    mainRender.set_shade_model(GL_FLAT)


    with open(annFile,'r') as fio:
        ann_str = json.load(fio)

    dome_dense_train = {}
    dome_dense_train['images'] = []
    dome_dense_train['annotations'] = []

    all_train_images = ann_str['images']


    iid = 100
    for img in all_train_images[::5]:
        #print(img)
        img_name = os.path.splitext(img['file_name'])[0]

        img_parts = img_name.split('_')
        frame_id = int(img_parts[-1])
        view_id = int(img_parts[-2])
        seq_name = '_'.join(img_parts[:-2])
        c_pklFile = os.path.join('/media/internal/domedb',seq_name,'hdPose3d_Adam_stage1','bodyAdam_{:08d}.pkl'.format(frame_id))
        c_calib = total_calibs[seq_name]
        cam_param = [cam for cam in c_calib if cam['node']==view_id][0]
        R = np.array(cam_param['R'])
        t = np.array(cam_param['t'])
        K = np.array(cam_param['K'])*1280.0/1920.0
        vt_all_cuda,vc_all,f_all,ids = paramfile_to_render_objects(c_pklFile)
        R_cuda = torch.tensor(R[None,:,:],dtype=torch.float32).cuda()
        t_cuda = torch.tensor(t[None,None,:,0],dtype=torch.float32).cuda()
        K_cuda = torch.tensor(K[None,:,:],dtype=torch.float32).cuda()

        vt_2d_cuda = geo_utils.projection_cuda(vt_all_cuda,K_cuda,R_cuda,t_cuda,1280,720)


        #projection
        color_img,depth_img = mainRender.render_obj(vt_all_cuda.cpu().numpy(),vc_all/255.0,None,f_all,R,t,K)
        color_img = color_img[:,:,:3]
        #channel 1. mask
        depth_img_cuda = torch.tensor(depth_img[None,None,:,:],dtype=torch.float32).cuda()
        vt_2d_grid = vt_2d_cuda[:,:,None,:2]

        depth_sample_cuda = torchF.grid_sample(depth_img_cuda,vt_2d_grid)
        depth_sample = depth_sample_cuda.cpu().numpy().transpose(0,2,1,3)[:,:,:,0]

        vt_2d = vt_2d_cuda.cpu().numpy()
        vis_diff = nlg.norm(depth_sample - vt_2d[:,:,2][:,:,None],ord=np.inf,axis=2).ravel()
        
        cAnns = []
        for cid,adam_id in enumerate(ids):
            c_vis_diff = vis_diff[18540*cid:18540*(cid+1)]
            c_vts_2d = vt_2d[0,18540*cid:18540*(cid+1),:]

            cmask = color_img[:,:,1] == adam_id+1
            binary_alpha = np.zeros((720,1280,3),dtype=np.uint8)
            binary_alpha[cmask,:] = 1
            binary_mask = binary_alpha[:,:,0]
            uvi_img_masked = color_img*binary_alpha
            cAnn = from_reprojection(uvi_img_masked,binary_mask,c_vts_2d,c_vis_diff)
            if cAnn is None:
                continue
            cAnn['keypoints']  = [0]*51
            cAnn['num_keypoints'] = 0
            cAnn['iscrowd'] = 0
            cAnn['category_id'] = 1
            cAnn['image_id'] = img['id']
            cAnn['id'] = cAnn['image_id']*100+adam_id
            cAnns.append(cAnn)
        iid +=1
        dome_dense_train['images'].append(copy.deepcopy(img))
        dome_dense_train['annotations'].extend(copy.deepcopy(cAnns))
        print(iid)
    with open('/media/posefs3b/Users/xiu/COCO/annotations/densepose_coco_2014_train.json') as fio:
        dp_coco = json.load(fio)
    
    cats = dp_coco['categories']
    dome_dense_train['categories'] = copy.deepcopy(cats)

    with open('tmp_dataset.json','w') as fio:
        json.dump(dome_dense_train,fio)
    
if __name__ == '__main__':
    pre_load_calibs()
    main_regen()
