import torch

def batch_rodrigues_torch(theta, transpose_r = False):
    #theta N*J x 3
    # batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    
    # quat N*Jx4
    return quat, quat2mat_torch(quat, transpose_r)

def quat2mat_torch(quat, transpose_r = False):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [N*J, 4] <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [N*J, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    if not transpose_r:
        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                              2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                              2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    else:
        rotMat = torch.stack([w2 + x2 - y2 - z2,    2*wz + 2*xy,    2*xz - 2*wy,
                            2*xy - 2*wz,   w2 - x2 + y2 - z2,  2*wx + 2*yz,
                            2*wy + 2*xz,    2*yz - 2*wx,    w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)

    return rotMat


def make_A_torch(N,R,t):

    R_homo = torch.cat([R,torch.zeros(N,1,3,dtype=torch.float32).cuda()],dim=1)
    t_homo = torch.cat([t.view(N,3,1), torch.ones(N,1,1,dtype=torch.float32).cuda()], dim = 1)
    return torch.cat([R_homo, t_homo], dim=2)