import scipy.sparse
import numpy as np
import sklearn.preprocessing
import torch
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



def projection(vt,K,R,t,imgw,imgh):
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
