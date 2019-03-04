import scipy.linalg
import numpy
import numpy.linalg
import numpy.random

def DLTmethod(Amat):
    _,_,V = scipy.linalg.svd(Amat)
    X = V[-1,:]
    X = X/X[-1]
    p3d = X[0:3]
    return p3d

def triangulate_onepoint_naive(pts,P_matrices):
#Input Nx2 pts, N cams
    nview = pts.shape[0]
    if nview<2:
        return numpy.array([0,0,0]),-1
    Amat = numpy.zeros((nview*2,4))
    for i in range(nview):
        Amat[2*i:2*i+2,:] = numpy.outer(pts[i,:],P_matrices[i,2,:]) - P_matrices[i,0:2,:]
    p3d = DLTmethod(Amat)
    rep_err = reproject_errors_multiview(p3d,P_matrices,pts)
    mean_err = numpy.mean(rep_err)
    if mean_err>50:
        p3d = None
    return p3d,mean_err

def triangulate_onepoint_RANSAC(pts,P_matrices,pts_prior):

    nview = pts.shape[0]
    Amat = numpy.zeros((nview,2,4)) 
    for i in range(nview):
        Amat[i,:,:] = numpy.outer(pts[i,:],P_matrices[i,2,:]) - P_matrices[i,0:2,:]
    #full whole Amat

    #get RANSAC params
    maxIter = 10
    minValid_num = 2
    minCons_num = 2
    inlier_thr = 10
    best_err = 1e3
    best_p3d = None
    for n_iter in range(maxIter):
        all_id = numpy.arange(nview)
        numpy.random.shuffle(all_id)
        est_id = all_id[:minCons_num]
        tst_id = all_id[minCons_num:]
        estAmat = Amat[est_id,:,:].reshape(-1,4)
        cp3d = DLTmethod(estAmat)
        rep_err_term = reproject_errors_multiview(cp3d,P_matrices[tst_id,:,:],pts[tst_id,:])
        tst_inliers = tst_id[rep_err_term<inlier_thr]
        #if enough inliers
        if len(tst_inliers) >= minValid_num:
            re_id = numpy.concatenate( (est_id, tst_inliers) )
            estAmat_re = Amat[re_id,:,:].reshape(-1,4)
            cp3d_re = DLTmethod(estAmat_re)
            err_re = reproject_errors_multiview(cp3d,P_matrices[re_id,:,:],pts[re_id,:])
            new_err_rep = numpy.mean(err_re)
            #new_err_prior = numpy.linalg.norm(cp3d_re - pts_prior)
            new_err = new_err_rep # + new_err_prior
            if new_err < best_err:
                best_err = new_err
                best_p3d = cp3d_re
    return best_p3d,best_err

def reproject_3Dpoint(p3d,P_matrix):
    #p3d as n*3, all data mat is store as nxp
    #P_matrix as 3*4 K*[R,t]
    p3dh = numpy.hstack((p3d,numpy.ones((p3d.shape[0],1))))
    p2dh = P_matrix.dot(p3dh.T)
    p2dh = p2dh[0:2,:]/p2dh[2,:]
    return p2dh.T

def reproject_errors_oneview(p3d,P_matrix,p2ds):
    p2drep = reproject_3Dpoint(p3d,P_matrix)
    err = numpy.linalg.norm(p2ds-p2drep,axis=1)
    return err

def reproject_errors_multiview(p3d,P_matrices,p2ds):
    p3dh = numpy.append(p3d,1)
    num_pts = p2ds.shape[0]
    rep2ds = numpy.zeros((num_pts,2))
    for vid in range(num_pts):
        p2dh = P_matrices[vid].dot(p3dh.T)
        rep2ds[vid,:] = p2dh[0:2]/p2dh[2]
    err = numpy.linalg.norm(rep2ds-p2ds,axis=1)
    return err


