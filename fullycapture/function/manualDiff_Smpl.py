import torch
import cPickle as pickle
class manualDiff_SMPL(object):
    def __init__(self,pkl_path):
        with open(pkl_path) as fio:
            dd  = pickle.load(fio)
        J_reg_ = dd['J_regressor']
        Bs_ = dd['shapedirs']
        self.dJ_dbeta =   torch.matmul(J_reg_,Bs_)

    
    def eval_with_Jacobian(theta,betas):
        return 0
        
# def manualDiff_SMPL(theta,betas):
#     a = 0