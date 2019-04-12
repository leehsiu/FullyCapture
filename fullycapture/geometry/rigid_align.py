import torch
import torch.nn
import numpy as np
import numpy.linalg as nlg

def Rmat_to_axis(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = np.zeros(3)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = np.arctan2(r, t-1)
    # Normalise the axis.
    axis = axis / r
    # Return the data.
    return axis, theta

def rigid_align_cpu(src,dest):
    assert(src.shape[1]==3)
    c_src = np.mean(src,axis=0)
    c_dest = np.mean(dest,axis=0)

    H_mat = (src - c_src).T.dot((dest-c_dest))
    U,_,Vh = nlg.svd(H_mat)

    R = (Vh.T).dot(U.T)
    if nlg.det(R) < 0:
        Vh[2,:] *= -1
        R = (Vh.T).dot(U.T)
    t = -R.dot(c_src.T) + c_dest.T
    #t = c_dest.T - c_src.T
    return R,t

def rigid_align_torch(src,dest):
    '''
    calculate the rigid transform between src and tar

    '''
    c_src = torch.mean(src,dim=0)[None,:]
    #print(c_src)
    c_dest = torch.mean(dest,dim=0)[None,:]
    H_mat = torch.mm((src - c_src).transpose(1,0),(dest-c_dest))
    U,S,V = torch.svd(H_mat)
    R = torch.mm(V,U.transpose(1,0))
    if torch.det(R) < 0:
        V[:,2] *= -1
        R = torch.mm(V,U.transpose(1,0))
    #print(R)
    t = - torch.mm(R,c_src.transpose(1,0)) + c_dest.transpose(1,0)
    
    return R,t