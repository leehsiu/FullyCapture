import cv2
import numpy as np

def undistort_verts(pts,K,distcoef):
    valid_ind = pts[:,0]>0
    dp_featuresf = pts[valid_ind,:-2:-1]/1.0
    vt = dp_featuresf[:,:,np.newaxis]
    vt = np.transpose(vt,(0,2,1))
    undist = cv2.undistortPoints(vt,K,distcoef)
    up = undist[:,0,:]
    up = np.hstack((up,np.ones((dp_featuresf.shape[0],1),dtype=up.dtype)))
    upL = np.transpose(K.dot(up.T))
    pts[valid_ind,:-2:-1] = upL[:,0:1]
    return pts

