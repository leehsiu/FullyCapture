import ctypes
from PIL import Image, ImageOps
import numpy as np
import cPickle as pickle
import copy

class TotalCLib(object):
	def __init__(self, lib_file='../../build/libtotalCops.so'):
		self.lib = ctypes.cdll.LoadLibrary(lib_file)
		self.lib.load_SMPLModelMale.argtypes = [ctypes.POINTER(ctypes.c_int), #f
											ctypes.POINTER(ctypes.c_double), #J_reg
											ctypes.POINTER(ctypes.c_int),
											ctypes.c_int,
											ctypes.POINTER(ctypes.c_int), #kintree
											ctypes.POINTER(ctypes.c_double), #U_
											ctypes.POINTER(ctypes.c_double), #mu_
											ctypes.POINTER(ctypes.c_double), #J_mu
											ctypes.POINTER(ctypes.c_double), #LBS_w
											ctypes.POINTER(ctypes.c_double), #coco_reg
											ctypes.POINTER(ctypes.c_int),
											ctypes.c_int] 
		self.lib.load_SMPLModelMale.restype = None

		self.lib.load_SMPLModelFemale.argtypes = [ctypes.POINTER(ctypes.c_int), #f
											ctypes.POINTER(ctypes.c_double), #J_reg
											ctypes.POINTER(ctypes.c_int),
											ctypes.c_int,
											ctypes.POINTER(ctypes.c_int), #kintree
											ctypes.POINTER(ctypes.c_double), #U_
											ctypes.POINTER(ctypes.c_double), #mu_
											ctypes.POINTER(ctypes.c_double), #J_mu
											ctypes.POINTER(ctypes.c_double), #LBS_w
											ctypes.POINTER(ctypes.c_double), #coco_reg
											ctypes.POINTER(ctypes.c_int),
											ctypes.c_int] 
		self.lib.load_SMPLModelFemale.restype = None

		self.lib.load_SMPLModelNeutral.argtypes = [ctypes.POINTER(ctypes.c_int), #f
											ctypes.POINTER(ctypes.c_double), #J_reg
											ctypes.POINTER(ctypes.c_int),
											ctypes.c_int,
											ctypes.POINTER(ctypes.c_int), #kintree
											ctypes.POINTER(ctypes.c_double), #U_
											ctypes.POINTER(ctypes.c_double), #mu_
											ctypes.POINTER(ctypes.c_double), #J_mu
											ctypes.POINTER(ctypes.c_double), #LBS_w
											ctypes.POINTER(ctypes.c_double), #coco_reg
											ctypes.POINTER(ctypes.c_int),
											ctypes.c_int] 
		self.lib.load_SMPLModelNeutral.restype = None


		self.lib.setup_fit_options.argtypes = [ctypes.c_double]*5
		self.lib.setup_fit_options.restype = None
        #extern "C" void smpl_fit_total_stage1(double* pose, double* coeff, double* trans, double* targetJoint, int reg_type, bool fit_shape, bool showiter);
		self.lib.smpl_fit_total_stage1.argtypes = [ctypes.POINTER(ctypes.c_double)]*4+[ctypes.c_int]*2+[ctypes.c_bool]*2
		self.lib.smpl_fit_total_stage1.restype = None

	def smpl_fit_stage1(self,target_joints,betas,pose,trans,reg_type,gender='male',fit_shape=True,show_iter=True):
		assert(target_joints.shape==(65,4))
		joints_c = target_joints.flatten().tolist()

		pose_c = (ctypes.c_double*156)()
		betas_c = (ctypes.c_double*10)()
		trans_c = (ctypes.c_double*3)()


		pose_c[:] = pose.flatten().tolist()
		betas_c[:] = betas.flatten().tolist()
		trans_c[:] = trans.flatten().tolist()

		gender_c = 0 if gender=='female' else 1

		self.lib.smpl_fit_total_stage1(pose_c,
							   		betas_c,
									trans_c,
									(ctypes.c_double*len(joints_c))(*joints_c),
									ctypes.c_int(reg_type),
									ctypes.c_int(gender_c),
									fit_shape,show_iter)

		betas_np = np.frombuffer(betas_c,float).copy()
		pose_np = np.frombuffer(pose_c,float).copy()
		trans_np = np.frombuffer(trans_c,float).copy()

		return betas_np,pose_np,trans_np


	def load_SMPLModel(self,modelpath,gender):

		with open(modelpath) as f:
			SMPLpkl = pickle.load(f)
		faces_ = SMPLpkl['f'].flatten().tolist()
		#faces_.astype(int,copy=False)        
		J_reg_ = SMPLpkl['J_regressor']
		J_reg_coo_ = J_reg_.tocoo()
		J_reg_val = J_reg_coo_.data.tolist()
		J_reg_ind = J_reg_coo_.row.tolist()+J_reg_coo_.col.tolist()
		J_reg_nnz = len(J_reg_val)
		U_ = SMPLpkl['shapedirs'].reshape(6890*3,10)
		U_ = np.array(U_)
		U_ = U_.flatten().tolist()
		kintree_table_ = SMPLpkl['kintree_table'].flatten().tolist()
		kintree_table_[0] = -1

		W_ = SMPLpkl['weights'].flatten().tolist()

		mu_ = SMPLpkl['v_template'].flatten().tolist()

		J_mu_ = SMPLpkl['J_regressor'].dot(SMPLpkl['v_template']).flatten().tolist()

		coco_reg_ = SMPLpkl['J_regressor_total']
		coco_reg_coo = coco_reg_.tocoo()
		coco_reg_val = coco_reg_coo.data.tolist()
		coco_reg_ind = coco_reg_coo.row.tolist()+coco_reg_coo.col.tolist()
		coco_reg_nnz = len(coco_reg_val)
		if gender=='male':
			self.lib.load_SMPLModelMale((ctypes.c_int*len(faces_))(*faces_),
									(ctypes.c_double*len(J_reg_val))(*J_reg_val),
									(ctypes.c_int*len(J_reg_ind))(*J_reg_ind),
									ctypes.c_int(J_reg_nnz),
									(ctypes.c_int*len(kintree_table_))(*kintree_table_),
									(ctypes.c_double*len(U_))(*U_),
									(ctypes.c_double*len(mu_))(*mu_),
									(ctypes.c_double*len(J_mu_))(*J_mu_),
									(ctypes.c_double*len(W_))(*W_),
									(ctypes.c_double*len(coco_reg_val))(*coco_reg_val),
									(ctypes.c_int*len(coco_reg_ind))(*coco_reg_ind),
									ctypes.c_int(coco_reg_nnz))
		elif gender=='female':
			self.lib.load_SMPLModelFemale((ctypes.c_int*len(faces_))(*faces_),
						(ctypes.c_double*len(J_reg_val))(*J_reg_val),
						(ctypes.c_int*len(J_reg_ind))(*J_reg_ind),
						ctypes.c_int(J_reg_nnz),
						(ctypes.c_int*len(kintree_table_))(*kintree_table_),
						(ctypes.c_double*len(U_))(*U_),
						(ctypes.c_double*len(mu_))(*mu_),
						(ctypes.c_double*len(J_mu_))(*J_mu_),
						(ctypes.c_double*len(W_))(*W_),
						(ctypes.c_double*len(coco_reg_val))(*coco_reg_val),
						(ctypes.c_int*len(coco_reg_ind))(*coco_reg_ind),
						ctypes.c_int(coco_reg_nnz))
		else:
			self.lib.load_SMPLModelNeutral((ctypes.c_int*len(faces_))(*faces_),
						(ctypes.c_double*len(J_reg_val))(*J_reg_val),
						(ctypes.c_int*len(J_reg_ind))(*J_reg_ind),
						ctypes.c_int(J_reg_nnz),
						(ctypes.c_int*len(kintree_table_))(*kintree_table_),
						(ctypes.c_double*len(U_))(*U_),
						(ctypes.c_double*len(mu_))(*mu_),
						(ctypes.c_double*len(J_mu_))(*J_mu_),
						(ctypes.c_double*len(W_))(*W_),
						(ctypes.c_double*len(coco_reg_val))(*coco_reg_val),
						(ctypes.c_int*len(coco_reg_ind))(*coco_reg_ind),
						ctypes.c_int(coco_reg_nnz))