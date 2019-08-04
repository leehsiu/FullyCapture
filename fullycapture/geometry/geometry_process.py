import scipy.sparse
import numpy as np
import cv2
import sklearn.preprocessing
import torch

def vertices_normals_torch(f,vertex_by_face,v):
    fNormal_u = v[f[:,1],:] - v[f[:,0],:]
    fNormal_v = v[f[:,2],:] - v[f[:,0],:]
    fNormal = torch.cross(fNormal_u,fNormal_v)
    fNormal_norm = torch.norm(fNormal, p = 2, dim = 1)

    fNormal = torch.div(fNormal,fNormal_norm)

    vNormal = torch.matmul(vertex_by_face,fNormal)
    vNormal_norm = torch.norm(vNormal,p=2,dim=1)
    vNormal = torch.div(vNormal,vNormal_norm)
    
    return vNormal


def vertices_normals(f,v):

	fNormal_u = v[f[:,1],:] - v[f[:,0],:]
	fNormal_v = v[f[:,2],:] - v[f[:,0],:]
	fNormal = np.cross(fNormal_u,fNormal_v)
	fNormal = sklearn.preprocessing.normalize(fNormal)
	
	vbyf_vid = f.flatten('F')
	vbyf_fid = np.arange(f.shape[0])
	vbyf_fid = np.concatenate((vbyf_fid,vbyf_fid,vbyf_fid))
	vbyf_val = np.ones(len(vbyf_vid))
	vbyf = scipy.sparse.coo_matrix((vbyf_val,(vbyf_vid,vbyf_fid)),shape=(v.shape[0],f.shape[0])).tocsr()
	
	vNormal = vbyf.dot(fNormal)

	vNormal = sklearn.preprocessing.normalize(vNormal)

	return vNormal


def projection_torch(vt,K,R,t,imgw,imgh):
    '''
    Input
        v: NxVx3 vertices
        K: Nx3x3 intrinsic 
        R: Nx3x3 camera rotation
        t: Nx3x1 camera translation
        imsize: Nx1 imgsize
    Output
        [u,v,z] in image
    '''
    eps = 1e-9
    #v = torch.matmul()
    vt = torch.matmul(vt,R.transpose(2,1)) + t
    x,y,z = vt[:,:,0],vt[:,:,1],vt[:,:,2]
    x_ = x / (z + eps)
    y_ = y / (z + eps)
    #no distortion
    vt = torch.stack([x_,y_,torch.ones_like(z)],dim=-1)
    vt = torch.matmul(vt,K.transpose(1,2))
    u,v = vt[:,:,0],vt[:,:,1]
    u = 2 * (u - imgw/2.) / imgw
    v = 2 * (v - imgh/2.) / imgh

    #normlize vt to [-1,1]

    vt = torch.stack([u,v,z],dim=-1)

    return vt


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

