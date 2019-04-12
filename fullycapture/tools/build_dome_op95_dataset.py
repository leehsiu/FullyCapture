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

adam_to_subdiv_path = '/home/xiul/workspace/pyopenpose/openpose/datasets/adam_to_subdiv.txt'
adam_to_subdiv = np.loadtxt(adam_to_subdiv_path)

op95_kps = scipy.io.loadmat('/home/xiul/workspace/PanopticDome/matlab/OP95_kps.mat')
adam_to_kp95 = scipy.io.loadmat('/home/xiul/workspace/PanopticDome/matlab/adam_to_op95_cls_start_with_1.mat')

adam_cls = adam_to_kp95['adam_cls'].ravel()
adam_weight = adam_to_kp95['adam_weight'].ravel()

vis_adam_cls = adam_to_kp95['vis_adam_cls'].ravel()
vis_adam_weight = adam_to_kp95['vis_adam_weight'].ravel()
vis_adam_inds = adam_to_kp95['vis_adam_inds'].ravel()

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



    kps_95_gt_x = []
    kps_95_gt_y = []
    kps_95_gt_w = []
    kps_count = 0
    for kps_id in range(1,96):
        find_inds, = np.where(np.logical_and(vis_adam_cls==kps_id,vis_2d<1e-1))
        if len(find_inds)<1:
            kps_95_gt_x.append(-1)
            kps_95_gt_y.append(-1)
            kps_95_gt_w.append(-1)
        else:
            max_inds = np.argmax(vis_adam_weight[find_inds])
            
            if vis_adam_weight[find_inds[max_inds]]<0.5:
                kps_95_gt_x.append(-1)
                kps_95_gt_y.append(-1)
                kps_95_gt_w.append(-1)
            else:
                kps_count += 1

                c_x = vts_2d[find_inds[max_inds],0]*640+640
                c_y = vts_2d[find_inds[max_inds],1]*360+360
                c_w = vis_adam_weight[find_inds[max_inds]]
                c_x_256 = (c_x - coco_bbox[0])/coco_bbox[2]*256.0
                c_y_256 = (c_y - coco_bbox[1])/coco_bbox[3]*256.0
                kps_95_gt_x.append(c_x_256)
                kps_95_gt_y.append(c_y_256)
                kps_95_gt_w.append(c_w)
    dp_unlabeled = np.zeros((256,256),dtype=np.uint8)
    c_unlabel_fortran = np.asfortranarray(dp_unlabeled)
    c_unlabel_mask = mask_util.encode(c_unlabel_fortran)


   #cAnn['dp_miss_code'] = c_unlabel_mask
    
    cAnno = {"segmentation":coco_segments,"dp_masks":coco_dp_masks_rle,"area":int(coco_area),"bbox":coco_bbox.tolist(),
            "dp_I":[],"dp_U":[],"dp_V":[],"dp_x":[],"dp_y":[],'op95_x':kps_95_gt_x,'op95_y':kps_95_gt_y,'op95_w':kps_95_gt_w,'dp_miss_code':c_unlabel_mask}
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


def processRecord_sparse(render_utils,c_record):

    frameId = c_record['frame_id']
    seqName = c_record['sequence']
    c_pklFile = os.path.join('/media/internal/domedb',seqName,'hdPose3d_Adam_stage1','bodyAdam_{:08d}.pkl'.format(frameId))
    c_calib = total_calibs[seqName]

    cam_param = c_calib[np.random.randint(len(c_calib))]
    R = np.array(cam_param['R'])
    t = np.array(cam_param['t'])
    K = np.array(cam_param['K'])*1280.0/1920.0

    #vt_all is in cuda
    vt_all_cuda,vc_all,f_all,ids,vis_inds = paramfile_to_render_objects(c_pklFile)

    R_cuda = torch.tensor(R[None,:,:],dtype=torch.float32).cuda()
    t_cuda = torch.tensor(t[None,None,:,0],dtype=torch.float32).cuda()
    K_cuda = torch.tensor(K[None,:,:],dtype=torch.float32).cuda()

    vt_2d_cuda = geo_utils.projection_cuda(vt_all_cuda[:,vis_inds,:],K_cuda,R_cuda,t_cuda,1280,720)

    #projection
    color_img,depth_img = render_utils.render_obj(vt_all_cuda.cpu().numpy(),vc_all/255.0,None,f_all,R,t,K)
    color_img = color_img[:,:,:3]
    #channel 1. mask
    depth_img_cuda = torch.tensor(depth_img[None,None,:,:],dtype=torch.float32).cuda()
    vt_2d_grid = vt_2d_cuda[:,:,None,:2]

    depth_sample_cuda = torchF.grid_sample(depth_img_cuda,vt_2d_grid)
    depth_sample = depth_sample_cuda.cpu().numpy().transpose(0,2,1,3)[:,:,:,0]

    vt_2d = vt_2d_cuda.cpu().numpy()
    vis_diff = nlg.norm(depth_sample - vt_2d[:,:,2][:,:,None],ord=np.inf,axis=2).ravel()

    
    #the visibility information is here.



    cAnns = []
    cImageInfo = None

    # cImgPath = '/media/posefs0c/panoptic/{}/hdImgs/00_{:02d}/00_{:02d}_{:08d}.jpg'.format(seqName,cam_param['node'],cam_param['node'],frameId)
    # if not os.path.isfile(cImgPath):
    #     return cAnns,cImageInfo

    # t0 = time.time()
    # cImg = cv2.imread(cImgPath)
    # if is_full_green(cImg):
    #     return cAnns,cImageInfo
    # t1 = time.time()

    #print('imread time {}'.format(t1-t0))

    for cid,adam_id in enumerate(ids):

        c_vis_diff = vis_diff[3522*cid:3522*(cid+1)]
        c_vts_2d = vt_2d[0,3522*cid:3522*(cid+1),:]

        cmask = color_img[:,:,1] == adam_id+1
        binary_alpha = np.zeros((720,1280,3),dtype=np.uint8)
        binary_alpha[cmask,:] = 1
        binary_mask = binary_alpha[:,:,0]
        uvi_img_masked = color_img*binary_alpha

        cAnn = from_reprojection(uvi_img_masked,binary_mask,c_vts_2d,c_vis_diff)

        cImageInfo = create_image_info(seqName,cam_param['node'],frameId)
        if cAnn is not None:
            cAnn['keypoints'] = []
            cAnn['num_keypoints'] = 0
            cAnn['iscrowd'] = 0
            cAnn['category_id'] = 1
            cAnn['image_id'] = cImageInfo['id']
            cAnn['id'] = cAnn['image_id']*100+adam_id
            cAnns.append(copy.deepcopy(cAnn))
    return cAnns,cImageInfo

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
        c_vis = vis_adam_inds + idx*adamWrapper.size[0]
        vis_inds.extend(c_vis.tolist())


    f_all = np.stack(f_all,axis=0)
    vc_all = np.stack(vc_all,axis=0)
    vis_inds = np.array(vis_inds,dtype=np.int32)

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

    return vt_all,vc_all,f_all,ids, vis_inds



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
    



def main_sparse(args):  
    # Init EGL 

    mainRender = EglRender(1280,720)
    mainRender.enable_light(False)
    mainRender.set_shade_model(GL_FLAT)

    #get shard list
    shard_list = args.shard_id
    print(shard_list)
    shard_ids = [int(ci) for ci in shard_list.split(',')]
    print(shard_ids)
    for shard_id in shard_ids:      
        with open('/media/posefs1b/Users/xiu/panopticdb/J_filter/instance_shard_{:08d}.json'.format(shard_id*65536)) as fio:
            shard_record = json.load(fio)        
        dome_dense_train = {}
        dome_dense_train['images'] = []
        dome_dense_train['annotations'] = []

        dome_dense_test = {}
        dome_dense_test['images'] = []
        dome_dense_test['annotations'] = []

        skip = args.skip
        for idx,c_record in enumerate(shard_record[::skip]):
            t0 = time.time()
            ann,img = processRecord_sparse(mainRender,c_record)
            t1 = time.time()

            if ann:
                seqname = c_record['sequence']
                if seqname in test_set:
                    dome_dense_test['annotations'] += copy.deepcopy(ann)
                    dome_dense_test['images'].append(copy.deepcopy(img))
                else:
                    dome_dense_train['annotations'] += copy.deepcopy(ann)
                    dome_dense_train['images'].append(copy.deepcopy(img))
            t1 = time.time()
            print('shard {} : {}/{} in {}'.format(shard_id,idx,65536/skip,t1-t0))

        with open('/media/posefs1b/Users/xiu/panopticdb/dome_op95_train_{:d}.json'.format(shard_id),'w') as fio:
            json.dump(dome_dense_train,fio)
        with open('/media/posefs1b/Users/xiu/panopticdb/dome_op95_test_{:d}.json'.format(shard_id),'w') as fio:
            json.dump(dome_dense_test,fio)
    
if __name__ == '__main__':
    args = parse_args()
    pre_load_calibs()
    main_sparse(args)
