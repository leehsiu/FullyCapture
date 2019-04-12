import sys
import torch
import numpy as np
import time
import totaldensify.optimizer.torch_smplify as smplify_util
from totaldensify.model.batch_adam_torch import AdamModelTorch
from totaldensify.vis.glut_viewer import glut_viewer
import cPickle as pickle
import totaldensify.data.dataIO as dataIO
import matplotlib.pyplot as plt
import totaldensify.vis.plot_vis as plot_vis
import neural_renderer as nr
import os.path
import cv2
c_map = plt.get_cmap('hsv')

fig, ax = plt.subplots()



def mannual_correct(inds_img):
    inds_dict_img = [10,7,4,9,8,6,3,11,5,2]
    inds_dict_target = [3,2,8,5,6,1,10,4,9,7]

    inds_img_c0 = inds_img[:, :, 0]
    mask_img = np.zeros(inds_img_c0.shape, inds_img_c0.dtype)
    for img_i,target_i in zip(inds_dict_img,inds_dict_target):
        mask_img[inds_img_c0==img_i] = target_i
    #re-arrange idx as 1,2,3 counting numbers
    mask_img = np.dstack((mask_img, mask_img, mask_img))

    return mask_img

def test_vis_densecoco(root_path,viewId,frameId):
    img_file = os.path.join(root_path,'images','00_{:02d}_{:08d}.jpg'.format(viewId,frameId))
    vts_file = os.path.join(root_path,'dp_vts_e2e','00_{:02d}_{:08d}_001.txt'.format(viewId,frameId))

    img = cv2.imread(img_file)[:,:,::-1]
    vts = np.loadtxt(vts_file)

    ax.imshow(img)
    ax.scatter(vts[:,1],vts[:,2],s=1,c='r')
    plt.show()
def test_load_data(img_path, root_path):
    fit_data = dataIO.prepare_data_total(root_path, img_path)
    n_body = len(fit_data['joints'])

    ax.imshow(fit_data['img'])
    mask_img = mannual_correct(fit_data['inds_img'])
    alpha_img = np.zeros((mask_img.shape[0],mask_img.shape[1]))
    alpha_img[mask_img[:,:,0]>0] = 1.0

    color_img = np.zeros(mask_img.shape,np.float32)
    for idx in range(n_body):
        color_img[mask_img[:,:,0]==(idx+1),:] = c_map(idx/(n_body+1.0))[:3]
    color_img = np.dstack((color_img,alpha_img[:,:,None]))

    ax.imshow(color_img, alpha=0.5)
    for idx,(jts,jtsw) in enumerate(zip(fit_data['joints'],fit_data['joints_weight'])):
        plot_vis.plot_coco25_joints(jts, jtsw, ax, c_map(idx/(n_body+1.0)))
    
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()


if __name__ == '__main__':
    # img_path = '/home/xiul/databag/denseFusion/images/run.jpg'
    # root_path = '/home/xiul/databag/denseFusion'
    # test_load_data(img_path, root_path)
    root_path = '/home/xiul/databag/dome_sptm/171204_pose6'
    test_vis_densecoco(root_path,2,500)