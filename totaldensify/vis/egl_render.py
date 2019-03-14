'''
This code has minimum part for opengl visualization, including floor visualization and mouse+keyboard control
'''
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.EGL import *
import numpy as np
import numpy.linalg
import numpy as np
import cPickle as pickle
import chumpy
import json
import argparse
import time
import glob
import os.path
import copy
import sklearn.preprocessing
import cv2
import scipy.io
import sys


class EglRender(object):
	def __init__(self,width,height):
		self.width = width
		self.height = height

		self.configAttribs = [
			EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
			EGL_BLUE_SIZE, 8,
			EGL_GREEN_SIZE, 8,
			EGL_RED_SIZE, 8,
			EGL_DEPTH_SIZE, 8,
			EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
			EGL_NONE]
		self.pbufferAttribs = [
			EGL_WIDTH, int(self.width),
			EGL_HEIGHT, int(self.height),
			EGL_NONE,]
		self.eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY)
		self.major = EGLint()
		self.minor = EGLint()
		eglInitialize(self.eglDpy, ctypes.byref(self.major), ctypes.byref(self.minor))
		#2. Select an appropriate configuration
		self.numConfigs = EGLint()
		self.eglCfg = EGLConfig()

		eglChooseConfig(self.eglDpy, self.configAttribs, ctypes.byref(self.eglCfg), 1, ctypes.byref(self.numConfigs))

		#3. Create a surface
		self.eglSurf = eglCreatePbufferSurface(self.eglDpy, self.eglCfg, self.pbufferAttribs)

	#// 4. Bind the API
		eglBindAPI(EGL_OPENGL_API)

	# // 5. Create a context and make it current
		self.eglCtx = eglCreateContext(self.eglDpy, self.eglCfg, EGL_NO_CONTEXT,
							  None)

		eglMakeCurrent(self.eglDpy, self.eglSurf, self.eglSurf, self.eglCtx)

		self.colorTex = None
		self.depthTex = None
		self.fbo = None
		self.depth_buffer = None
		self.vn_buffer = None
		self.vt_buffer = None
		self.vc_buffer = None
		self.f_buffer = None
		self.lighting = False

		self.glContentInit()
		print('Headless renderer initialized')
	
	def enable_light(self,enabled):
		self.lighting = enabled
	def glContentInit(self):
		glClearColor(0.0, 0.0, 0.0, 0.0)
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		# Enable depth testing
		glShadeModel(GL_SMOOTH)
		#glDisable(GL_LIGHTING)
		glEnable(GL_LIGHTING)
		glEnable(GL_LIGHT0) 
		glEnable(GL_ALPHA_TEST)
		glEnable(GL_TEXTURE_2D)

		self.colorTex = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.colorTex)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width,
					self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
		glBindTexture(GL_TEXTURE_2D, 0)


		self.depthTex = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.depthTex)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, self.width,
					self.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
		glBindTexture(GL_TEXTURE_2D, 0)

		self.fbo = glGenFramebuffers(1)
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		glFramebufferTexture2D(
			GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.colorTex, 0)
		self.depth_buffer = glGenRenderbuffers(1)
		glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer)
		glRenderbufferStorage(
			GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
		glFramebufferRenderbuffer(
			GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_buffer)
		glFramebufferTexture2D(
			GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depthTex, 0)
		glBindFramebuffer(GL_FRAMEBUFFER, 0)

		self.vn_buffer = glGenBuffers(1)
		self.vt_buffer = glGenBuffers(1)
		self.vc_buffer = glGenBuffers(1)
		self.f_buffer = glGenBuffers(1)
		self.z_far = 5000.
		self.z_near = 1.

	def bind_buffers(self,v,vc,vn,f):
		if v is not None:
			v_c = v.flatten().tolist()
			glBindBuffer(GL_ARRAY_BUFFER, self.vt_buffer)
			glBufferData(GL_ARRAY_BUFFER, len(v_c)*sizeof(ctypes.c_float),
						(ctypes.c_float*len(v_c))(*v_c), GL_STATIC_DRAW)
		if vn is not None:
			vn_c = vn.flatten().tolist()
			glBindBuffer(GL_ARRAY_BUFFER, self.vn_buffer)
			glBufferData(GL_ARRAY_BUFFER, len(vn_c)*sizeof(ctypes.c_float),
						(ctypes.c_float*len(vn_c))(*vn_c), GL_STATIC_DRAW)
		if vc is not None:
			vc_c = vc.flatten().tolist()
			glBindBuffer(GL_ARRAY_BUFFER, self.vc_buffer)
			glBufferData(GL_ARRAY_BUFFER, len(vc_c)*sizeof(ctypes.c_float),
						(ctypes.c_float*len(vc_c))(*vc_c), GL_STATIC_DRAW)
		if f is not None:
			f_c = f.astype(np.int).flatten().tolist()
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.f_buffer)
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ctypes.c_int) * len(f_c),
							(ctypes.c_int * len(f_c))(*f_c), GL_STATIC_DRAW)

	def render_buffers(self,elem_num):

		glEnableVertexAttribArray(0)
		glEnableVertexAttribArray(3)
		glBindBuffer(GL_ARRAY_BUFFER, self.vt_buffer)
		glVertexAttribPointer(
					0,                                # attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_FALSE,                         # normalized?
					0,  # / stride
					None                         # array buffer offset
				)
		glBindBuffer(GL_ARRAY_BUFFER, self.vc_buffer)
		glVertexAttribPointer(
					3,                                # 1 attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_FALSE,                         # normalized?
					0,  # / stride
					None                         # array buffer offset
				)

		glEnableVertexAttribArray(2)
		glBindBuffer(GL_ARRAY_BUFFER, self.vn_buffer)
		glVertexAttribPointer(
					2,                                # attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_TRUE,                         # normalized?
					0,  # / stride
					None                         # array buffer offset
				)
		
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.f_buffer)
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
		glDrawElements(GL_TRIANGLES, elem_num*3, GL_UNSIGNED_INT, None)
		
		glDisableVertexAttribArray(2)
		glDisableVertexAttribArray(0)
		glDisableVertexAttribArray(3)

	def set_camera(self,R,t,K):
		invR = np.array(R)
		invT = np.array(t)
		camMatrix = np.hstack((invR, invT))
		#denotes camera matrix, [R|t]
		camMatrix = np.vstack((camMatrix, [0, 0, 0, 1]))
		camK = np.array(K)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()        

		#!!! Notice that this is the correct version.
		ProjM = np.zeros((4,4))
		ProjM[0,0] = 2*camK[0,0]/self.width
		ProjM[0,2] = (self.width - 2*camK[0,2])/self.width
		ProjM[1,1] = -2*camK[1,1]/self.height
		ProjM[1,2] = -(-self.height+2*camK[1,2])/self.height
		ProjM[2,2] = (-self.z_far-self.z_near)/(self.z_far-self.z_near)
		ProjM[2,3] = -2*self.z_far*self.z_near/(self.z_far - self.z_near)
		ProjM[3,2] = -1



		glLoadMatrixd(ProjM.T)





		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)
		glMultMatrixd(camMatrix.T)

	def render_obj(self,v,vc,vn,f,R,t,K):
		#prepare buffers.
		self.bind_buffers(v,vc,vn,f)
		elem_num = f.shape[0]

		glBindFramebuffer(GL_FRAMEBUFFER,self.fbo)
		#render once
		#render
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glClearColor(0,0,0,0)
		# set up camerasetCamera(g_camid)
		self.set_camera(R,t,K)
		if self.lighting:
			glEnable(GL_LIGHTING)
		else:
			glDisable(GL_LIGHTING)
		self.render_buffers(elem_num)

		glBindTexture(GL_TEXTURE_2D,self.colorTex)
		color_str = glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE)
		color_data = np.fromstring(color_str,dtype=np.uint8)
		color_img = np.reshape(color_data,(self.height,self.width,4))
		#color_img = np.flip(color_img,0)

# depthSample = 2.0 * depthSample - 1.0;
#     float zLinear = 2.0 * zNear * zFar / (zFar + zNear - depthSample * (zFar - zNear));

		glBindTexture(GL_TEXTURE_2D,0)

		glBindTexture(GL_TEXTURE_2D,self.depthTex)
		depth_str = glGetTexImage(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT,GL_FLOAT)
		depth_data = np.fromstring(depth_str,dtype=np.float32)
		depth_z = np.reshape(depth_data,(self.height,self.width))
		depth_z = 2.0*depth_z - 1.0
		depth_img = 2.0*self.z_near*self.z_far/(self.z_far + self.z_near - depth_z *(self.z_far - self.z_near))
		#depth_img = np.flip(depth_img,0)
		glBindTexture(GL_TEXTURE_2D,0)

		return color_img,depth_img
	def __exit__(self):
		#terminate EGL when finished
		print('ended, free up the display')
		eglTerminate(self.eglDpy)

