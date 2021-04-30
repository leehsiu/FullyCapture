import torch
import torch.nn.functional as F
from rodrigues import *

def rodrigues(theta):
    # theta N*J x 3
    # batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    # quat N*Jx4
    return quat2mat(quat)


def quat2mat(quat):

    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def make_T(N, R, t):
    R_homo = torch.cat(
        [R, torch.zeros([N, 1, 3], dtype=torch.float32, device=R.device)], dim=1)
    t_homo = torch.cat([t.view(N, 3, 1), torch.ones(
        [N, 1, 1], dtype=torch.float32, device=t.device)], dim=1)
    return torch.cat([R_homo, t_homo], dim=2)


def kinematric_transform(Rs, Js, kintree_parent):
    # Rs NxJx3x3
    # Js NxJx3
    n_batch_ = Rs.shape[0]
    n_J_ = Rs.shape[1]
    # Js NxJx3 -> NxJx3x1
    Js = torch.unsqueeze(Js, -1)
    outT = n_J_*[None]

    outT[0] = make_T(n_batch_, Rs[:, 0, :, :], Js[:, 0])

    for idj in range(1, kintree_parent.shape[0]):
        ipar = kintree_parent[idj]
        J_local = Js[:, idj] - Js[:, ipar]
        T_local = make_T(n_batch_, Rs[:, idj], J_local)
        outT[idj] = torch.matmul(outT[ipar], T_local)

    outT = torch.stack(outT, dim=1)
    outJ = outT[:, :, :3, 3]
    Js = F.pad(Js, [0, 0, 0, 1])
    outT_0 = torch.matmul(outT, Js)
    outT_0 = F.pad(outT_0, [3, 0, 0, 0, 0, 0, 0, 0])
    outT = outT - outT_0
    return outJ, outT


def lbs(verts, weights, theta, Js, kintree_parent):
    # input shape

    #verts  : NxVx3
    #weights: NxVxJ
    #theta  : NxJx3
    #Js     : NxJx3
    #kintree_parent : NxJx1

    n_batch_ = theta.shape[0]

    theta = theta.view(-1, 3)
    Rs = rodrigues(theta)
    Rs = Rs.view(n_batch_, -1, 3, 3)

    Js_out, Ts_out = kinematric_transform(Rs, Js, kintree_parent)
    Ts_vec = Ts_out.view(n_batch_, -1, 16)

    T = torch.matmul(weights, Ts_vec).view(n_batch_, -1, 4, 4)

    verts = torch.cat([verts, torch.ones(
        [n_batch_, verts.shape[1], 1], dtype=torch.float32, device=verts.device)], dim=2)
    verts = torch.matmul(T, torch.unsqueeze(verts, -1))
    verts = verts[:, :, :3, 0]

    return verts
