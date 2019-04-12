import matplotlib.pyplot as plt
import numpy as np

def plot_coco25_joints_3d(joints,weight,ax,c):
    # X = joints[:,0]
    # Y = joints[:,1]
    coco25_parents = [1,1,1,2,3,1,5,6,1,8,9,10,8,12,13,0,0,15,16,14,19,14,11,22,11]
    coco25_ids = range(25)
    for i in coco25_ids:
        if weight[i]>0.1 and weight[coco25_parents[i]]>0.1:
            ax.plot(joints[[coco25_ids[i],coco25_parents[i]],0],joints[[coco25_ids[i],coco25_parents[i]],1],joints[[coco25_ids[i],coco25_parents[i]],2],c=c)

def plot_total_joints_3d(joints,weight,ax,c):
    # X = joints[:,0]
    # Y = joints[:,1]
    coco25_parents = [1,1,1,2,3,1,5,6,1,8,9,10,8,12,13,0,0,15,16,14,19,14,11,22,11]

    rhand_parents = [4,1+24,2+24,3+24,4,5+24,6+24,7+24,4,9+24,10+24,11+24,4,13+24,14+24,15+24,4,17+24,18+24,19+24]
    lhand_parents =[7,1+44,2+44,3+44,7,5+44,6+44,7+44,7,9+44,10+44,11+44,7,13+44,14+44,15+44,7,17+44,18+44,19+44]

    total_parents =coco25_parents + rhand_parents+lhand_parents
    total_ids = range(65)
    for i in total_ids:
        if weight[i]>0.1 and weight[total_parents[i]]>0.1:
            ax.plot(joints[[i,total_parents[i]],0],joints[[i,total_parents[i]],1],joints[[i,total_parents[i]],2],c=c,marker='o',markersize=2)


def plot_3d_axis_equal(joints,ax):
    x_max = joints[:,0].max()
    x_min = joints[:,0].min()
    y_max = joints[:,1].max()
    y_min = joints[:,1].min()
    z_max = joints[:,2].max()
    z_min = joints[:,2].min()
    plot_radius = np.array([x_max-x_min,y_max-y_min,z_max-z_min]).max()/1.5
    ax.set_xlim((x_max+x_min)/2.0 - plot_radius,(x_max+x_min)/2.0+plot_radius)
    ax.set_ylim((y_max+y_min)/2.0 - plot_radius,(y_max+y_min)/2.0+plot_radius)
    ax.set_zlim((z_max+z_min)/2.0 - plot_radius,(z_max+z_min)/2.0+plot_radius)



def plot_coco25_joints(joints,weight,ax,c):
    # X = joints[:,0]
    # Y = joints[:,1]
    coco25_parents = [1,1,1,2,3,1,5,6,1,8,9,10,8,12,13,0,0,15,16,14,19,14,11,22,11]
    coco25_ids = range(25)
    for i in coco25_ids:
        if weight[i]>0.1 and weight[coco25_parents[i]]>0.1:
            ax.plot(joints[[coco25_ids[i],coco25_parents[i]],0],joints[[coco25_ids[i],coco25_parents[i]],1],c=c)
    # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    # mid_x = (X.max()+X.min()) * 0.5
    # mid_y = (Y.max()+Y.min()) * 0.5
    # mid_z = (Z.max()+Z.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)


def make_overlay(img0,img1,alpha):
	img0f = img0.astype(np.float32)
	img1f = img1.astype(np.float32)
	alphaf = alpha.astype(np.float32)/255.0
	alphaf = alphaf[:,:,None].repeat(3,axis=2)
	img0_on_1 = img0f * alphaf + (1-alphaf) * img1f
	return img0_on_1.astype(np.uint8)
