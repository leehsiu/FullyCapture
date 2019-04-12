from totaldensify.model.batch_smpl import SmplModel
import numpy as np
import scipy.sparse
if __name__=='__main__':
    smplMaleCpu = SmplModel('../../models/SMPLH_male_with_total_reg.pkl')
    smpl_v = smplMaleCpu.v_template
    smpl_f = smplMaleCpu.f
    vbyf_vid = smpl_f.flatten('F')
    vbyf_fid = np.arange(smpl_f.shape[0])
    vbyf_fid = np.concatenate((vbyf_fid,vbyf_fid,vbyf_fid))
    vbyf_val = np.ones(len(vbyf_vid))
    vbyf = scipy.sparse.coo_matrix((vbyf_val,(vbyf_vid,vbyf_fid)),shape=(smpl_v.shape[0],smpl_f.shape[0])).todense()
    v_patten = []
    v_patten_len = []
    for vid in range(smpl_v.shape[0]):
        f_from_v = vbyf[vid,:].flatten()
        f_id = np.where(f_from_v>0)[1]
        v_ids = smpl_f[f_id,:].flatten().tolist()
        c_patten = set(v_ids) - set([vid])

        c_patten = [vid] + list(c_patten)
        c_len = len(c_patten)
        if c_len<8:
            c_patten = c_patten + c_patten

        c_patten = c_patten[:8]
        v_patten.append(c_patten)
        v_patten_len.append(len(c_patten))
    v_patten = np.array(v_patten)

    np.savetxt('../../models/smpl_v_pattern.txt',v_patten)