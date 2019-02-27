import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from totaldensify.model.batch_adam_torch import AdamModelTorch
from totaldensify.vis.glut_viewer import glut_viewer
#import cv2
#from multivariate_normal_internal import *
#from torch.distributions.multivariate_normal import *


class BodyRecon:
	def __init__(self):
		self.mseLoss = torch.nn.MSELoss().cuda()
		id_torso_coco25 = [2, 5, 9, 12]
		weight_coco25 = np.ones((1, 25, 3)) * 0.5
		weight_coco25[:, id_torso_coco25,:] = 1
		self.weight_coco25_cu = torch.tensor(weight_coco25).cuda()


	def smplify3d_sgd_mannual(self, totalModel, j3d_t, init_param, n_iter, reg_type):
		n_batch = j3d_t.shape[0]

		betas_cu = torch.tensor(
			init_param['betas'], requires_grad=True, device='cuda').cuda()
		thetas_cu = torch.tensor(
			init_param['thetas'], requires_grad=True, device='cuda').cuda()
		j3d_t_cu = torch.tensor(j3d_t).cuda()
		j3d_t_cu = j3d_t_cu.permute(0, 2, 1)

		mask_torso_cu = self.mask_torso_coco25_cu.repeat(n_batch, 1, 1)

		assert(reg_type == 'coco25')
		# optimizer = torch.optim.Adam(
		# 	[{'params':betas_cu},{'params':}, lr=0.1).cuda()
		# #0. fit torso only

		for i in range(n_iter):
			_, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
			j3d_pred = j3d_pred.permute(0, 2, 1)
			j3d_pred = j3d_pred * mask_torso_cu
			j3d_t_cu_cur = j3d_t_cu * mask_torso_cu
			loss = self.mseLoss(j3d_t_cu_cur, j3d_pred)
			loss.backward()
			betas_cu.data -= 1e-1 * betas_cu.grad.data
			thetas_cu.data -= 1e-1 * thetas_cu.grad.data
			betas_cu.grad.zero_()
			thetas_cu.grad.zero_()
			print('loss in iter [{}/{}] : {}'.format(i, n_iter, loss.item()))

		for i in range(n_iter*0):
			_, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
			j3d_pred = j3d_pred.permute(0, 2, 1)
			loss = self.mseLoss(j3d_t_cu, j3d_pred)
			loss.backward()
			betas_cu.data -= 1e-2 * betas_cu.grad.data
			thetas_cu.data -= 1e-2 * thetas_cu.grad.data
			betas_cu.grad.zero_()
			thetas_cu.grad.zero_()
			print('loss in iter [{}/{}] : {}'.format(i, n_iter, loss.item()))

		return betas_cu.detach().cpu().numpy(), thetas_cu.detach().cpu().numpy()

	def smplify3d_adam(self,totalModel,j3d_t,init_param,n_iter,reg_type):

		n_batch = j3d_t.shape[0]
		betas_cu = torch.tensor(
			init_param['betas'], requires_grad=True, device='cuda').cuda()
		thetas_cu = torch.tensor(
			init_param['thetas'], requires_grad=True, device='cuda').cuda()
		trans_cu = torch.tensor(
			init_param['trans'],requires_grad=True, device = 'cuda').cuda()
		thetas_cu_zeros = torch.zeros_like(thetas_cu).cuda()
		betas_cu_zeros = torch.zeros_like(betas_cu).cuda()

		weight_cu = self.weight_coco25_cu.repeat(n_batch, 1, 1)


		j3d_t_cu = torch.tensor(j3d_t).cuda()

		optimizer = torch.optim.Adam([{'params':betas_cu},
									  {'params':thetas_cu},
									  {'params':trans_cu}], lr=0.5)
		# #0. fit torso only
		for i in range(n_iter):
			_, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
			j3d_pred = j3d_pred + trans_cu
			j3d_pred = j3d_pred * weight_cu
			j3d_t_cu_n = j3d_t_cu * weight_cu

			loss = self.mseLoss(j3d_t_cu_n, j3d_pred)
			loss_norm = self.mseLoss(thetas_cu,thetas_cu_zeros)
			loss_beta = self.mseLoss(betas_cu,betas_cu_zeros)
			loss_total = 0.1*loss + 10.0*loss_norm + 1*loss_beta
			optimizer.zero_grad()	
			loss_total.backward()
			optimizer.step()
			print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

		for i in range(n_iter):
			_, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
			j3d_pred = j3d_pred + trans_cu
			j3d_pred = j3d_pred * weight_cu
			j3d_t_cu_n = j3d_t_cu * weight_cu

			loss = self.mseLoss(j3d_t_cu_n, j3d_pred)
			loss_norm = self.mseLoss(thetas_cu,thetas_cu_zeros)
			loss_beta = self.mseLoss(betas_cu,betas_cu_zeros)
			loss_total = 0.1*loss + 1*loss_norm + 1*loss_beta
			optimizer.zero_grad()	
			loss_total.backward()
			optimizer.step()
			print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

		for i in range(n_iter):
			_, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
			j3d_pred = j3d_pred + trans_cu
			j3d_pred = j3d_pred * weight_cu
			j3d_t_cu_n = j3d_t_cu * weight_cu

			loss = self.mseLoss(j3d_t_cu_n, j3d_pred)
			loss_norm = self.mseLoss(thetas_cu,thetas_cu_zeros)
			loss_beta = self.mseLoss(betas_cu,betas_cu_zeros)
			loss_total = 0.1*loss + 0.1*loss_norm + 1*loss_beta
			optimizer.zero_grad()	
			loss_total.backward()
			optimizer.step()
			print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

		return betas_cu.detach().cpu().numpy(),thetas_cu.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()
	#def __body_prior(self,)
	#internal function
	def __smplify3d(self, totalModel, j3d_t, init_param, n_iter, reg_type):
		betas_cu = torch.tensor(
			init_param['betas'], requires_grad=True, device='cuda').cuda()
		thetas_cu = torch.tensor(
			init_param['thetas'], requires_grad=True, device='cuda').cuda()
		j3d_t_cu = torch.tensor(j3d_t).cuda()

		for i in range(n_iter):
			_, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)

			loss = self.mseLoss(j3d_t_cu, j3d_pred)
			loss.backward()
			betas_cu.data -= 1e-2 * betas_cu.grad.data
			thetas_cu.data -= 1e-2 * thetas_cu.grad.data
			betas_cu.grad.zero_()
			thetas_cu.grad.zero_()
			#print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss.item()))
		return betas_cu.detach().cpu().numpy(), thetas_cu.detach().cpu().numpy()

	def Densify3D(self, totalModel, j3d_t, v3d_t, init_param, n_iter, reg_type):
		betas_cu = torch.tensor(
			init_param['betas'], requires_grad=True, device='cuda').cuda()
		thetas_cu = torch.tensor(
			init_param['thetas'], requires_grad=True, device='cuda').cuda()
		j3d_t_cu = torch.tensor(j3d_t).cuda()
		v3d_t_cu = torch.tensor(v3d_t).cuda()
		j3d_t_cu = j3d_t_cu.permute(0, 2, 1)
		v3d_t_cu = v3d_t_cu.permute(0, 2, 1)

		for i in range(n_iter):
			v3d_pred, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
			j3d_pred = j3d_pred.permute(0, 2, 1)
			v3d_pred = v3d_pred.permute(0, 2, 1)
			loss_j = self.mseLoss(j3d_t_cu, j3d_pred)
			loss_v = self.mseLoss(v3d_t_cu, v3d_pred)
			loss = loss_j + 0.5*loss_v
			loss.backward()
			betas_cu.data -= 1e-2 * betas_cu.grad.data
			thetas_cu.data -= 1e-2 * thetas_cu.grad.data
			betas_cu.grad.zero_()
			thetas_cu.grad.zero_()
			print('loss in iter [{}/{}] : {}'.format(i, n_iter, loss.item()))
		return betas_cu.detach().cpu().numpy(), thetas_cu.detach().cpu().numpy()

	def SMPLify2D(self, totalModel, j2d_t, init_param, n_iter):

		return 0

	def Densify2D(self, totalModel, init_param, feature_map, n_iter):
		#TODO
		#feature map is a NxCxWxH volume

		#generate target full feature map using totalModel and init_param

		#optimize over the whole param

		return 0


	def __projection(self,vt,K,R,t,imsize):
		'''
		Input
			v: NxVx3 vertices
			K: Nx3x3 intrinsic 
			R: Nx3x3 camera rotation
			t: Nx3x1 camera translation
			imsize: Nx1 imgsize
		Output
		'''
		eps = 1e-9
		#v = torch.matmul()
		vt = torch.matmul(vt,R.transpose(2,1)) + t
		x,y,z = vt[:,:,0],vt[:,:,1],vt[:,:,2]
		x_ = x / (z + eps)
		y_ = y / (z + eps)
		#no distortion
		vt = torch.stack([x,y,torch.ones_like(z)],dim=-1)
		vt = torch.matmul(vt,K.transpose(1,2))
		u,v = vt[:,:,0],vt[:,:,1]
		u = 2 * (u - imsize/2.) / imsize
		v = 2 * (v - imsize/2.) / imsize

		#normlize vt to [-1,1]

		vt = torch.stack([u,v,z],dim=-1)

		return vt

def test_coco25():
	model_path = '/home/xiul/workspace/TotalDensify/models/adamModel_with_coco25_reg.pkl'
	totalModel = AdamModelTorch(pkl_path=model_path, reg_type='coco25')
	model_path = '/home/xiul/databag/dome_sptm/171204_pose6/gt_pkl'

	import glob
	import os.path
	model_pkls = glob.glob(os.path.join(model_path, '*.pkl'))
	model_pkls.sort()

	#model_pkls = model_pkls[:100]
	n_batch = len(model_pkls)

	betas = np.zeros((n_batch, 30))
	thetas = np.zeros((n_batch, 62, 3))
	trans = np.zeros((n_batch,1,3))

	betas_zero = np.zeros((n_batch, 30))
	thetas_zero = np.zeros((n_batch, 62, 3))
	trans_zero = np.zeros((n_batch,1,3))

	import cPickle as pickle
	for i_file, model_file in enumerate(model_pkls):
		with open(model_file) as fio:
			cdd = pickle.load(fio)
		betas[i_file] = cdd[0]['betas']
		thetas[i_file] = np.reshape(cdd[0]['pose'], (-1, 3))
		trans[i_file] = cdd[0]['trans']

	# glRender.init_GLUT(sys.argv)

	recon = BodyRecon()

	betas_cu = torch.tensor(betas).cuda()
	thetas_cu = torch.tensor(thetas).cuda()
	trans_cu = torch.tensor(trans).cuda()
	v3d_pred, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type='coco25')

	v3d_pred = v3d_pred + trans_cu
	j3d_pred = j3d_pred + trans_cu
	# n_iter = 100
	# t0 = time.time()
	# for i in range(n_iter):
	#     v3d_pred, j3d_pred = totalModel(betas_cu,thetas_cu,reg_type='coco25')
	# t1 = time.time()
	# param_betas_cu = torch.tensor(betas,requires_grad=True,device='cuda').cuda()
	# param_thetas_cu = torch.tensor(thetas,requires_grad=True,device='cuda').cuda()
	j3d_t = j3d_pred.detach().cpu().numpy()
	n_iter = 100
	t0 = time.time()
	init_param = {'betas': betas_zero, 'thetas': thetas_zero,'trans':trans_zero}
	gt_param = {'betas':betas,'thetas':thetas,'trans':trans}

	# betas_n, thetas_n = recon.SMPLify3D(
	# 	totalModel, j3d_t, init_param, n_iter, 'coco25')
	# betas_n, thetas_n,trans_n= recon.smplify3d_adam(
	# 	totalModel, j3d_t, init_param, n_iter, 'coco25')
	betas_n, thetas_n,trans_n= recon.smplify3d_adam(
		totalModel, j3d_t, init_param, n_iter, 'coco25')
	
	
	t1 = time.time()

	print('Time for each forward pass:{} sec'.format((t1 - t0)/(n_iter*n_batch)))

	betas_cu_n = torch.tensor(betas_n).cuda()
	thetas_cu_n = torch.tensor(thetas_n).cuda()
	trans_cu_n = torch.tensor(trans_n).cuda()

	v3d_n, j3d_n = totalModel(betas_cu_n, thetas_cu_n, reg_type='coco25')
	v3d_n = v3d_n + trans_cu_n
	j3d_n = j3d_n + trans_cu_n

	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'

	glRender = glut_viewer(1280, 720)
	glRender.loadCalibs(calibFile)

	v3d_pred_cpu = v3d_n.detach().cpu().numpy()
	j3d_pred_cpu = j3d_n.detach().cpu().numpy()

	glRender.load_data(v3d_pred_cpu, j3d_pred_cpu)
	glRender.init_GLUT(sys.argv)



def test_torch():
	N = 100
	V = 6890
	vt = np.zeros((N,V,3))
	t = np.zeros((N,1,3))
	R = np.ones((N,3,3))
	vt_cu = torch.tensor(vt).cuda()
	R_cu = torch.tensor(R).cuda()
	t_cu = torch.tensor(t).cuda()

	vt_n = torch.matmul(vt_cu,R_cu.transpose(2,1)) + t_cu

	print(vt_n.shape)


def example_data():
	model_path = '../../models/adamModel_with_coco25_reg.pkl'
	totalModel = AdamModelTorch(pkl_path=model_path, reg_type='coco25')
	model_path = '/home/xiul/databag/dome_sptm/171204_pose6/gt_pkl'

	import glob
	import os.path
	model_pkls = glob.glob(os.path.join(model_path, '*.pkl'))
	model_pkls.sort()

	model_pkls = model_pkls[:2]
	#model_pkls = model_pkls + model_pkls
	n_batch = len(model_pkls)

	betas = np.zeros((n_batch, 30))
	thetas = np.zeros((n_batch, 62, 3))

	betas_zero = np.zeros((n_batch, 30))
	thetas_zero = np.zeros((n_batch, 62, 3))

	import cPickle as pickle
	for i_file, model_file in enumerate(model_pkls):
		with open(model_file) as fio:
			cdd = pickle.load(fio)
		betas[i_file] = cdd[0]['betas']
		thetas[i_file] = np.reshape(cdd[0]['pose'], (-1, 3))


if __name__ == '__main__':
	#test_adam62()
	test_coco25()
	#test_torch()