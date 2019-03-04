import sys
import json
import math
import time
import numpy as np
import numpy.linalg as nlg
import torch
import torch.nn as nn
import cPickle as pickle

class BodyPrior(object):
    def __init__(self,pkl_path,n_gauss=8,n_prefix=3):
        self.n_gauss = n_gauss
        self.n_prefix = n_prefix
        with open(pkl_path) as fio:
            gmm = pickle.load(fio)

        inv_covars = np.array([nlg.inv(cov) for cov in gmm['covars']])
        means = np.array(gmm['means'])
        inv_chols = np.array([nlg.cholesky(inv_cov) for inv_cov in inv_covars])
        sqrdets_covars = np.array([(np.sqrt(np.linalg.det(c)))for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)
        weights = np.array(gmm['weights'] / (const*(sqrdets_covars / sqrdets_covars.min())))

        self.inv_chols_cu = []
        self.means_cu = []
        for inv_chol in inv_chols:
            self.inv_chols_cu.append(torch.tensor(inv_chol[None,:,:]))
        for mean in means:
            self.means_cu.append(torch.tensor(mean[None,:]))
    def __call__(self,theta_cu):
        n_batch = theta_cu.shape[0]
        self.inv_chols_batch_cu = []
        self.means_batch_cu = []
        for chols in self.inv_chols_cu:
            self.inv_chols_batch_cu.append(chols.repeat(n_batch,1,1))

        self.log_likes = []
if __name__=='__main__':
    prior = BodyPrior('/home/xiul/workspace/PanopticDome/models/gmm_08.pkl')
