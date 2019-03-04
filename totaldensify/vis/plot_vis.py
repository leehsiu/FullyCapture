import matplotlib.pyplot as plt


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

