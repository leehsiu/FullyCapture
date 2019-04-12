import sys
import torch
import torch.nn
import numpy as np
import time
#import totaldensify.optimizer.torch_smplify as smplify_util
from totaldensify.model.batch_smpl_torch import SmplModelTorch
from totaldensify.model.batch_smpl import SmplModel
#from totaldensify.vis.glut_viewer import glut_viewer
import cPickle as pickle
import totaldensify.data.dataIO as dataIO
import matplotlib.pyplot as plt
import totaldensify.vis.plot_vis as plot_vis
import neural_renderer as nr
c_map = plt.get_cmap('hsv')

fig, ax = plt.subplots()

def projection(vt,K,R,t,imsize):
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
	u = 2 * (u - imsize/2.) / imsize
	v = 2 * (v - imsize/2.) / imsize

	#normlize vt to [-1,1]

	vt = torch.stack([u,v,z],dim=-1)

	return vt

def smplify2d_adam(totalModel,j2d_t,init_param,n_iter,reg_type):


	n_batch = j2d_t.shape[0]

	betas_cu = torch.tensor(
		init_param['betas'],requires_grad=True, device='cuda').cuda()
	thetas_cu = torch.tensor(
		init_param['theta'],requires_grad=True, device='cuda').cuda()
	trans_cu = torch.tensor(
		init_param['trans'],requires_grad=True, device = 'cuda').cuda()
	cam_K_cu = torch.tensor(init_param['K']).cuda()
	cam_R_cu = torch.tensor(init_param['R']).cuda()
	cam_t_cu = torch.tensor(init_param['t']).cuda()
	weights = torch.tensor(init_param['weight']).cuda()
	img_size = init_param['img_size']


	thetas_cu_zeros = torch.zeros_like(thetas_cu).cuda()
	betas_cu_zeros = torch.zeros_like(betas_cu).cuda()

	#weight_cu = self.weight_coco25_cu.repeat(n_batch, 1, 1)

	j2d_t = torch.tensor(j2d_t).cuda() 


	optimizer = torch.optim.Adam([{'params':betas_cu},
									{'params':thetas_cu},
									{'params':trans_cu}], lr=0.1)
	l2loss = torch.nn.MSELoss()
	

	for i in range(n_iter):
		_, j3d_pred = totalModel(betas_cu, thetas_cu, reg_type)
		j3d_pred = j3d_pred /100.0+ trans_cu
		#j3d_pred = j3d_pred / 100.0

		j2d_pred = projection(j3d_pred,cam_K_cu,cam_R_cu,cam_t_cu,img_size)[:,:,:2]
		loss = l2loss(j2d_pred*weights, j2d_t*weights)
		loss_norm = l2loss(thetas_cu,thetas_cu_zeros)
		loss_beta = l2loss(betas_cu,betas_cu_zeros)
		loss_total = loss + 0.001*loss_norm + 1*loss_beta
		optimizer.zero_grad()	
		loss_total.backward()
		optimizer.step()
		print('loss in iter [{}/{}] : {}'.format(i,n_iter,loss_total.item()))

	return betas_cu.detach().cpu().numpy(),thetas_cu.detach().cpu().numpy(),trans_cu.detach().cpu().numpy()



def test_load_data_multi(img_path,root_path):
	
	smpl_path = '/home/xiul/workspace/TotalDensify/models/smplModel_with_coco25_reg.pkl'
	smplModelGPU = SmplModelTorch(pkl_path=smpl_path,reg_type='coco25')
	smplModelCPU = SmplModel(pkl_path=smpl_path,reg_type='coco25')

	fit_data = dataIO.prepare_data_total(root_path,img_path)
	# print(fit_data)
	#ax.imshow(fit_data['img'])
	
	# plot_vis.plot_coco25_joints(fit_data['joints'],fit_data['joints_weight'],ax,'r')
	# plt.show()
	j2d_tar = fit_data['joints']
	weight_tar = fit_data['joints_weight']
	for w in weight_tar:
		w[19:] = 0

	weight_tar = weight_tar[:,:,None]
	weight_tar = np.repeat(weight_tar,2,axis=2)
	# betas_init_cu = torch.tensor(fit_data['betas_init'][None,:].astype(np.float64)).cuda()
	# pose_init_cu = torch.tensor(fit_data['pose_init'].reshape(24,3)[None,:,:].astype(np.float64)).cuda()
	# trans_init_cu = torch.tensor(fit_data['trans_init'][None,:]).cuda()

	# vts,jts = smplModelGPU(betas_init_cu,pose_init_cu,reg_type='coco25')
	# vts = vts/100.0
	# jts = jts/100.0
	# vts += trans_init_cu
	# jts += trans_init_cu



	K = np.eye(3).astype(np.float32)
	R = np.eye(3).astype(np.float32)
	t = np.zeros((1,3)).astype(np.float32)
	#cam = fit_data['cam'][0]

	#print(fit_data['joints'].shape
	num_body = len(fit_data['cam'])

	img_size = fit_data['img'].shape[1]
	img_height = fit_data['img'].shape[0]

	cam = [img_size,img_size/2,img_height/2]
	K[0,0] = cam[0]
	K[1,1] = cam[0]
	K[0,2] = cam[1]
	K[1,2] = cam[2]
	K[2,2] = 1
	K = K[None,:,:]
	R = R[None,:,:]
	t = t[None,:,:]


	init_param = {'betas':np.array(fit_data['betas_init']).astype(np.float32),
				'theta':np.array(fit_data['pose_init']).reshape(-1,24,3).astype(np.float32),
				'trans':np.array(fit_data['trans_init'])[:,None,:].astype(np.float32),
				'K':K,
				'R':R,
				't':t,
				'img_size':img_size,
				'weight':weight_tar}

	#j2d_tar[:,:,0] = 2 * (j2d_tar[:,:,0] - img_size/2.) / img_size
	j2d_tar = 2 * (j2d_tar - img_size/2.) / img_size

	print(np.array(fit_data['trans_init']).shape)
	fit = False


	if not fit:
		betas_init_cu = torch.tensor(init_param['betas']).cuda()
		pose_init_cu = torch.tensor(init_param['theta']).cuda()
		trans_init_cu = torch.tensor(init_param['trans']).cuda()
	else:
		betas,thetas,trans = smplify2d_adam(smplModelGPU,j2d_tar,init_param,100,'coco25')
		betas_init_cu = torch.tensor(betas).cuda()
		pose_init_cu = torch.tensor(thetas).cuda()
		trans_init_cu = torch.tensor(trans).cuda()

	vts,jts = smplModelGPU(betas_init_cu,pose_init_cu,reg_type='coco25')
	vts = vts/100.0
	jts = jts/100.0
	vts += trans_init_cu
	jts += trans_init_cu

	#def projection(vt,K,R,t,imsize):
	proj_2d = projection(jts,torch.tensor(K).cuda(),torch.tensor(R).cuda(),torch.tensor(t).cuda(),img_size)
	proj_2d_cpu = proj_2d.detach().cpu().numpy()*img_size/2 + img_size/2

	ax.imshow(fit_data['img'])

	# 	#j2d_tar

	rendered = nr.Renderer(image_size=img_size).cuda()
	rendered.light_intensity_directional = 0.0
	rendered.light_intensity_ambient = 1.0

	#model_faces = smplModelCPU.f[None,:,:]
	num_vertices = smplModelCPU.v_template.shape[0]
	model_faces = smplModelCPU.f
	for i in range(1,num_body):
		model_faces = np.vstack((model_faces,smplModelCPU.f+i*num_vertices))
	print(model_faces.shape)
	

	f_num = smplModelCPU.f.shape[0]

	model_faces_cu = torch.tensor(model_faces[None,:,:].astype(np.int32),dtype=torch.int32).cuda()

	tex_np = np.zeros((1,model_faces_cu.shape[1],2,2,2,3))
	for i in range(num_body):
		idx = i
		# tex_np[0,i*f_num:(i+1)*f_num,0,0,0,:] = c_map(idx/11.0)[:3]
		# tex_np[0,i*f_num:(i+1)*f_num,0,0,1,:] = c_map(idx/11.0)[:3]
		# tex_np[0,i*f_num:(i+1)*f_num,0,1,0,:] = c_map(idx/11.0)[:3]
		# tex_np[0,i*f_num:(i+1)*f_num,0,1,1,:] = c_map(idx/11.0)[:3]
		# tex_np[0,i*f_num:(i+1)*f_num,1,0,0,:] = c_map(idx/11.0)[:3]
		# tex_np[0,i*f_num:(i+1)*f_num,1,0,1,:] = c_map(idx/11.0)[:3]
		# tex_np[0,i*f_num:(i+1)*f_num,1,1,0,:] = c_map(idx/11.0)[:3]
		# tex_np[0,i*f_num:(i+1)*f_num,1,1,1,:] = c_map(idx/11.0)[:3]
		tex_np[0,i*f_num:(i+1)*f_num,:] = c_map(idx/11.0)[:3]
		#tex_np[0,i*f_num:(i+1)*f_num,:] = [i*0.01+0.01,i*0.01+0.01,i*0.01+0.01]
		
		
	tex = torch.tensor(tex_np,dtype=torch.float32).cuda()
	K_cu = torch.tensor(K,dtype=torch.float32).cuda()
	R_cu = torch.tensor(R,dtype=torch.float32).cuda()
	t_cu = torch.tensor(t,dtype=torch.float32).cuda()
	vts_float = vts.view(-1,3)
	vts_float = vts_float.type(torch.float32)[None,:,:]
	print(vts_float.shape)
	print(model_faces_cu.shape)
	images = rendered(vts_float,model_faces_cu,tex,None,K_cu,R_cu,t_cu,None,img_size)
	image_cpu = images.detach().cpu().numpy()[0].transpose((1,2,0))
	image_cpu = np.flip(image_cpu,0)
	image_cpu = image_cpu[:img_height,:,:]
	alpha_cpu = np.ones((image_cpu.shape[0],image_cpu.shape[1]))
	sum_color = np.sum(image_cpu,axis=2)

	alpha_cpu[sum_color<=1e-3] = 0
	rgba_image = np.dstack((image_cpu,alpha_cpu[:,:,None]))
	ax.imshow(rgba_image,alpha=0.5)
	#ax[1].imshow(image_cpu)
	plot_w = fit_data['joints_weight'][0]
	plot_w[19:] = 0
	#ax.scatter(man_u,man_v,s=5)
	for idx,proj in enumerate(proj_2d_cpu):
		plot_w = fit_data['joints_weight'][0]
		plot_w[19:] = 0
		plot_vis.plot_coco25_joints(proj[:,:2],plot_w,ax,c_map(idx/11.0))

	# for j2d,plot_w in zip(j2d_tar,fit_data['joints_weight']):
	# 	plot_vis.plot_coco25_joints(j2d*img_size/2+img_size/2,plot_w,ax,'g')
	# print(images.shape)
	# image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
	# print(image.shape)
	# img_uint8 = (255*image).astype(np.uint8)
	# ax.imshow(img_uint8)
	ax.set_xticks(())
	ax.set_yticks(())
	plt.show()
	
	
		
def test_load_data(img_path,root_path):
	smpl_path = '/home/xiul/workspace/TotalDensify/models/smplModel_with_coco25_reg.pkl'
	smplModelGPU = SmplModelTorch(pkl_path=smpl_path,reg_type='coco25')
	smplModelCPU = SmplModel(pkl_path=smpl_path,reg_type='coco25')
	model_faces = smplModelCPU.f[None,:,:]
	model_faces_cu = torch.tensor(model_faces.astype(np.int32),dtype=torch.int32).cuda()

	fit_data = dataIO.prepare_data_total(root_path,img_path)
	# print(fit_data)
	#ax.imshow(fit_data['img'])
	# plot_vis.plot_coco25_joints(fit_data['joints'],fit_data['joints_weight'],ax,'r')
	# plt.show()
	j2d_tar = fit_data['joints'][0][None,:]
	# betas_init_cu = torch.tensor(fit_data['betas_init'][None,:].astype(np.float64)).cuda()
	# pose_init_cu = torch.tensor(fit_data['pose_init'].reshape(24,3)[None,:,:].astype(np.float64)).cuda()
	# trans_init_cu = torch.tensor(fit_data['trans_init'][None,:]).cuda()

	# vts,jts = smplModelGPU(betas_init_cu,pose_init_cu,reg_type='coco25')
	# vts = vts/100.0
	# jts = jts/100.0
	# vts += trans_init_cu
	# jts += trans_init_cu



	K = np.eye(3)
	R = np.eye(3)
	t = np.zeros((1,3))
	#cam = fit_data['cam'][0]

	print(fit_data['joints'].shape)
	print(len(fit_data['cam']))

	img_size = fit_data['img'].shape[1]
	img_height = fit_data['img'].shape[0]

	cam = [img_size,img_size/2,img_height/2]
	K[0,0] = cam[0]
	K[1,1] = cam[0]
	K[0,2] = cam[1]
	K[1,2] = cam[2]
	K[2,2] = 1
	K = K[None,:,:]
	R = R[None,:,:]
	t = t[None,:,:]


	init_param = {'betas':fit_data['betas_init'][0][None,:].astype(np.float64),
				'theta':fit_data['pose_init'][0].reshape(24,3)[None,:,:].astype(np.float64),
				'trans':fit_data['trans_init'][0][None,:],
				'K':K,
				'R':R,
				't':t,
				'img_size':img_size}

	#j2d_tar[:,:,0] = 2 * (j2d_tar[:,:,0] - img_size/2.) / img_size
	j2d_tar = 2 * (j2d_tar - img_size/2.) / img_size


	fit = True


	if not fit:
		betas_init_cu = torch.tensor(init_param['betas']).cuda()
		pose_init_cu = torch.tensor(init_param['theta']).cuda()
		trans_init_cu = torch.tensor(init_param['trans']).cuda()
	else:
		betas,thetas,trans = smplify2d_adam(smplModelGPU,j2d_tar,init_param,100,'coco25')
		betas_init_cu = torch.tensor(betas).cuda()
		pose_init_cu = torch.tensor(thetas).cuda()
		trans_init_cu = torch.tensor(trans).cuda()

	vts,jts = smplModelGPU(betas_init_cu,pose_init_cu,reg_type='coco25')
	vts = vts/100.0
	jts = jts/100.0
	vts += trans_init_cu
	jts += trans_init_cu

	#def projection(vt,K,R,t,imsize):
	proj_2d = projection(jts,torch.tensor(K).cuda(),torch.tensor(R).cuda(),torch.tensor(t).cuda(),img_size)
	proj_2d_cpu = proj_2d.detach().cpu().numpy()[0]*img_size/2 + img_size/2

	ax.imshow(fit_data['img'])
	plot_w = fit_data['joints_weight'][0]
	plot_w[19:] = 0
	#ax.scatter(man_u,man_v,s=5)
	plot_vis.plot_coco25_joints(proj_2d_cpu[:,:2],plot_w,ax,'r')
	plot_vis.plot_coco25_joints(j2d_tar[0]*img_size/2+img_size/2,plot_w,ax,'g')
	# 	#j2d_tar


	rendered = nr.Renderer(image_size=img_size).cuda()
	rendered.light_intensity_directional = 0.0
	rendered.light_intensity_ambient = 1.0

	tex = torch.ones(1, model_faces_cu.shape[1], 2, 2, 2, 3, dtype=torch.float32).cuda()


	K_cu = torch.tensor(K,dtype=torch.float32).cuda()
	R_cu = torch.tensor(R,dtype=torch.float32).cuda()
	t_cu = torch.tensor(t,dtype=torch.float32).cuda()
	vts_render = vts.reshape(-1,3)[None,:,:]

	images = rendered(vts_render.type(torch.float32),model_faces_cu,tex,None,K_cu,R_cu,t_cu,None,img_size)

	image_cpu = images.detach().cpu().numpy()[0].transpose((1,2,0))
	image_cpu = np.flip(image_cpu,0)
	image_cpu = image_cpu[:img_height,:,:]
	ax.imshow(image_cpu,alpha=0.5)
#	ax[1].imshow(image_cpu)
	
	# print(images.shape)
	# image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
	# print(image.shape)
	# img_uint8 = (255*image).astype(np.uint8)
	# ax.imshow(img_uint8)
	plt.show()
	
	
	
	#writer.append_data((255*image).astype(np.uint8))

if __name__=='__main__':
	img_path = '/home/xiul/databag/denseFusion/images/run.jpg'
	root_path = '/home/xiul/databag/denseFusion'
	test_load_data_multi(img_path,root_path)