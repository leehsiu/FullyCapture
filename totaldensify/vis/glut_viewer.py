'''
This code has minimum part for opengl visualization, including floor visualization and mouse+keyboard control
'''
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

#import sys, json, math
import numpy as np
import numpy.linalg
from PIL import Image, ImageOps
import numpy as np
import cPickle as pickle
import scipy.sparse
import json
import argparse
import threading
import time
import glob
from totaldensify.model.batch_adam import ADAM
from totaldensify.model.batch_smpl import SMPL
import copy
import sklearn.preprocessing
from collections import deque

g_ambientLight = (0.35, 0.35, 0.35, 1.0)
g_diffuseLight = (0.75, 0.75, 0.75, 0.7)
g_specular = (1.0, 1.0, 1.0, 1.0)


adamPath = '/media/posefs3b/Users/xiu/PanopticDome/models/adamModel_lbs_200.pkl'


def VerticesNormals(f,v):

	fNormal_u = v[f[:,1],:] - v[f[:,0],:]
	fNormal_v = v[f[:,2],:] - v[f[:,0],:]
	fNormal = np.cross(fNormal_u,fNormal_v)
	fNormal = sklearn.preprocessing.normalize(fNormal)
	
	vbyf_vid = f.flatten('F')
	vbyf_fid = np.arange(f.shape[0])
	vbyf_fid = np.concatenate((vbyf_fid,vbyf_fid,vbyf_fid))
	vbyf_val = np.ones(len(vbyf_vid))
	vbyf = scipy.sparse.coo_matrix((vbyf_val,(vbyf_vid,vbyf_fid)),shape=(v.shape[0],f.shape[0])).tocsr()

	vNormal = vbyf.dot(fNormal)

	vNormal = sklearn.preprocessing.normalize(vNormal)

	return vNormal

class glut_viewer:
	def __init__(self,width,height):
		self.viewDist = 50
		self.width = width
		self.height = height
		self.zNear = 1.0
		self.zFar = 2000.0
		self.xMousePosPrev = 0
		self.yMousePosPrev = 0
		self.xTrans = 0
		self.yTrans = 0
		self.zTrans = 800
		self.xRot = 60
		self.yRot = -40
		self.zRot = 0
		self.viewMode = 'free'
		self.frameId = 0
		self.camId = deque('00',maxlen=2)
		self.adamWrapper = ADAM(pkl_path=adamPath)
		self.coco25_parents = [1,1,1,2,3,1,5,6,1,8,9,10,8,12,13,0,0,15,16,14,19,14,11,22,11]
		self.mouseAction = ""

		self.meshs = []
		self.joints = []
		self.frameNum = 0

	def init_GLUT(self,argv):
		glutInitDisplayMode(GLUT_RGB|GLUT_DOUBLE|GLUT_DEPTH)
		glutInitWindowPosition(100,100)
		glutInitWindowSize(self.width,self.height)
		glutInit(argv)
		glutCreateWindow("Panoptic Studio Viewer")
		self.init_GL()
		glutReshapeFunc(self.reshapeCallback)
		glutDisplayFunc(self.mainloop)
		glutKeyboardFunc(self.keyboardCallback)
		glutMouseFunc(self.mouseCallback)
		glutMotionFunc(self.motionCallback)
		glutSpecialFunc(self.specialkeysCallback)
		glutIdleFunc(self.idlefuncCallback)
		glutMainLoop()

	def specialkeysCallback(self,key,x,y):
		if key == GLUT_KEY_RIGHT:
			cFrameId = self.frameId
			cFrameId += 1
			if cFrameId==self.frameNum:
				cFrameId = 0
			self.frameId = cFrameId
		if key == GLUT_KEY_LEFT:
			cFrameId = self.frameId
			cFrameId -= 1
			if cFrameId <0:
				cFrameId = 0
			self.frameId = cFrameId

	def easyCylinder(self,v1,v2,dim,color):
		glColor4f(color[0], color[1], color[2], color[3])
		v2r = v2 - v1
		z = np.array([0.0, 0.0, 1.0])
		# the rotation axis is the cross product between Z and v2r
		ax = np.cross(z, v2r)
		import math
		l = math.sqrt(np.dot(v2r, v2r))
		# get the angle using a dot product
		angle = 180.0 / math.pi * math.acos(np.dot(z, v2r) / l)

		glPushMatrix()
		glTranslatef(v1[0], v1[1], v1[2])
		
		#print "The cylinder between %s and %s has angle %f and axis %s\n" % (v1, v2, angle, ax)
		glRotatef(angle, ax[0], ax[1], ax[2])
		glutSolidCylinder(dim / 10.0, l, 20, 20)
		glPopMatrix()

	def load_data(self,mesh_v,joints):
		self.frameNum = mesh_v.shape[0]
		self.joints = joints

		for i in range(self.frameNum):
			c_f = self.adamWrapper.f
			c_vn = VerticesNormals(c_f,mesh_v[i])
			self.meshs.append({'v':mesh_v[i],'vn':c_vn,'f':c_f})
		
	def init_GL(self):
		glClearColor(0.5, 0.5, 0.5, 0.0)
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glShadeModel(GL_SMOOTH)
		glEnable(GL_LIGHTING)

		glEnable(GL_ALPHA_TEST)
		glEnable(GL_TEXTURE_2D)
		glEnable(GL_LIGHT0) 
		glLightfv(GL_LIGHT0,GL_POSITION,[1,0,0,0])
		glEnable(GL_LIGHT1) 
		glLightfv(GL_LIGHT1,GL_POSITION,[0,1,0,0])
		glEnable(GL_LIGHT2) 
		glLightfv(GL_LIGHT2,GL_POSITION,[0,0,1,0])

		glEnable(GL_COLOR_MATERIAL)
		glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)


		self.colorTex = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.colorTex)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width,
					self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
		glBindTexture(GL_TEXTURE_2D, 0)

		self.bgTex = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.bgTex)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width,
					self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
		glBindTexture(GL_TEXTURE_2D, 0)
	
		self.fbo = glGenFramebuffers(1)
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		glFramebufferTexture2D(
			GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.colorTex, 0)
		self.depthBuffer = glGenRenderbuffers(1)
		glBindRenderbuffer(GL_RENDERBUFFER, self.depthBuffer)
		glRenderbufferStorage(
			GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
		glFramebufferRenderbuffer(
			GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depthBuffer)

		glBindFramebuffer(GL_FRAMEBUFFER, 0)

		self.numPolygon = 0
		self.vn_buffers = glGenBuffers(1)
		self.v_buffers = glGenBuffers(1)
		self.f_buffers = glGenBuffers(1)

	def renderDomeFloor(self):

		glDisable(GL_LIGHTING)
		gridNum = 10
		width = 50
		g_floorCenter = np.array([0,0,0])
		g_floorAxis1 = np.array([1,0,0])
		g_floorAxis2 = np.array([0,0,1])
		origin = g_floorCenter - g_floorAxis1*(width*gridNum/2 ) - g_floorAxis2*(width*gridNum/2)
		axis1 =  g_floorAxis1 * width
		axis2 =  g_floorAxis2 * width
		for y in range(gridNum+1):
			for x in range(gridNum+1):
				if (x+y) % 2 ==0:
					glColor(1.0,1.0,1.0,1.0) #white
				else:
					glColor(0.7,0.7,0.7,1) #grey
				p1 = origin + axis1*x + axis2*y
				p2 = p1+ axis1
				p3 = p1+ axis2
				p4 = p1+ axis1 + axis2
				glBegin(GL_QUADS)
				glVertex3f(   p1[0], p1[1], p1[2])
				glVertex3f(   p2[0], p2[1], p2[2])
				glVertex3f(   p4[0], p4[1], p4[2])
				glVertex3f(   p3[0], p3[1], p3[2])
				glEnd()

	def loadCalibs(self,calib_file):
		with open(calib_file) as f:
			rawCalibs = json.load(f)
		cameras = rawCalibs['cameras']
		allPanel = map(lambda x:x['panel'],cameras)
		hdCamIndices = [i for i,x in enumerate(allPanel) if x==0]
		self.hdCams = [cameras[i] for i in hdCamIndices]

	def setCamera(self,camid):
		if camid>=len(self.hdCams):
			camid = 0
		cam = self.hdCams[camid]
		invR = np.array(cam['R'])
		invT = np.array(cam['t'])
		camMatrix = np.hstack((invR, invT))
		# denotes camera matrix, [R|t]
		camMatrix = np.vstack((camMatrix, [0, 0, 0, 1]))
		#camMatrix = numpy.linalg.inv(camMatrix)
		K = np.array(cam['K'])
		#K = K.flatten()
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		Kscale = cam['resolution'][0]*1.0/self.width
		K = K/Kscale
		ProjM = np.zeros((4,4))
		ProjM[0,0] = 2*K[0,0]/self.width
		ProjM[0,2] = (self.width - 2*K[0,2])/self.width
		ProjM[1,1] = 2*K[1,1]/self.height
		ProjM[1,2] = (-self.height+2*K[1,2])/self.height
		ProjM[2,2] = (-self.zFar-self.zNear)/(self.zFar-self.zNear)
		ProjM[2,3] = -2*self.zFar*self.zNear/(self.zFar-self.zNear)
		ProjM[3,2] = -1

		glLoadMatrixd(ProjM.T)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)
		glMultMatrixd(camMatrix.T)

	def lookAt(self):
		if self.viewMode=='free':
			glMatrixMode(GL_PROJECTION)
			glLoadIdentity()
			cam = self.hdCams[0]
			invR = np.array(cam['R'])
			invT = np.array(cam['t'])
			camMatrix = np.hstack((invR,invT))
			camMatrix = np.vstack((camMatrix,[0,0,0,1]))
			K = np.array(cam['K'])
			fy = K[1,1]
			cy = K[1,2]
			fov = 360*np.arctan2(cy,fy)/np.pi
			gluPerspective(fov, float(self.width)/float(self.height),self.zNear,self.zFar)
			glMatrixMode(GL_MODELVIEW)
			glLoadIdentity()
			gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)
			self.setFreeView()
		else:
			camidlist = ''.join(self.camId)
			camid = int(camidlist)
			self.setCamera(camid)
	def renderPoints(self,v,colors):
	

		cvts = v.flatten()

		glBindBuffer(GL_ARRAY_BUFFER, self.v_buffers)
		glBufferData(GL_ARRAY_BUFFER, len(cvts)*sizeof(ctypes.c_float),
						(ctypes.c_float*len(cvts))(*cvts), GL_STATIC_DRAW)

		glEnableVertexAttribArray(0)
		glBindBuffer(GL_ARRAY_BUFFER, self.v_buffers)
		glVertexAttribPointer(
					0,                                # attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_FALSE,                         # normalized?
								0,  # / stride
					None                         # array buffer offset
				)
		glDrawElements(GL_POINTS, 18540*3, GL_FLOAT, None)

	def renderObj(self,v,vns,cinds,colors):
	
		self.numPolygon = cinds.shape[0]

		cvts = v.flatten()
		cvns = vns.flatten()
		cinds = cinds.flatten()

		glBindBuffer(GL_ARRAY_BUFFER, self.vn_buffers)
		glBufferData(GL_ARRAY_BUFFER, len(cvns)*sizeof(ctypes.c_float),
						(ctypes.c_float*len(cvns))(*cvns), GL_STATIC_DRAW)

		glBindBuffer(GL_ARRAY_BUFFER, self.v_buffers)
		glBufferData(GL_ARRAY_BUFFER, len(cvts)*sizeof(ctypes.c_float),
						(ctypes.c_float*len(cvts))(*cvts), GL_STATIC_DRAW)
		cinds = cinds.astype(np.int)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.f_buffers)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ctypes.c_uint) * len(cinds),
						(ctypes.c_uint * len(cinds))(*cinds), GL_STATIC_DRAW)
		

		glEnable (GL_BLEND)                                                                                     
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		#glBlendFunc(GL_ONE,GL_ONE)
		smpl_color = colors
		ambientFskeleton = 0.9
		diffuseFskeleton = 0.9
		specularFskeleton = 0.9
		smpl_shiness = 5.0
		smpl_ambient = [ambientFskeleton * smpl_color[0], ambientFskeleton *
					smpl_color[1], ambientFskeleton * smpl_color[2], 1.0]
		smpl_diffuse = [diffuseFskeleton * smpl_color[0], diffuseFskeleton *
					smpl_color[1], diffuseFskeleton * smpl_color[2], 1.0]
		smpl_spectular = [specularFskeleton * smpl_color[0], specularFskeleton *
					smpl_color[1], specularFskeleton * smpl_color[2], 1.0]

		#glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
		glPushMatrix()
		glEnable(GL_CULL_FACE)
		glEnable(GL_LIGHTING)
		glColor4f(smpl_color[0], smpl_color[1], smpl_color[2],smpl_color[3])
		glMaterialfv(GL_FRONT, GL_AMBIENT, smpl_ambient)
		glMaterialfv(GL_FRONT, GL_DIFFUSE, smpl_diffuse)
		glMaterialfv(GL_FRONT, GL_SPECULAR, smpl_spectular)
		glMaterialf(GL_FRONT, GL_SHININESS, smpl_shiness)
		glLineWidth(.5)
		glEnableVertexAttribArray(2)
		glBindBuffer(GL_ARRAY_BUFFER, self.vn_buffers)
		glVertexAttribPointer(
					2,                                # attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_TRUE,                         # normalized?
								0,  # / stride
					None                         # array buffer offset
				)
		glEnableVertexAttribArray(0)
		glBindBuffer(GL_ARRAY_BUFFER, self.v_buffers)
		glVertexAttribPointer(
					0,                                # attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_FALSE,                         # normalized?
								0,  # / stride
					None                         # array buffer offset
				)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.f_buffers)
		#glDrawElements(GL_TRIANGLES, fnum*3, GL_UNSIGNED_INT, None)
		glPolygonMode(GL_FRONT, GL_FILL)
		glDrawElements(GL_TRIANGLES, self.numPolygon*3, GL_UNSIGNED_INT, None)
		glPolygonMode(GL_FRONT, GL_FILL)
		glPopMatrix()		

	def renderJoints(self,Jtr,colors):
	
		for cid,pid in enumerate(self.coco25_parents):
			if cid==pid:
				continue	
			if Jtr[cid,3]>0.1 and Jtr[pid,3]>0.1:
				self.easyCylinder(Jtr[cid,0:3],Jtr[pid,0:3],15,[1.0,0.0,1.0,1])		
	

	def render_cameras(self):
		glDisable(GL_LIGHTING)
		sz = 10
		for cam in self.hdCams:
			invR = np.array(cam['R'])
			invT = np.array(cam['t'])
			camMatrix = np.hstack((invR,invT))
			camMatrix = np.vstack((camMatrix,[0,0,0,1]))
			camMatrix = numpy.linalg.inv(camMatrix)
			K = np.array(cam['K'])
			fx = K[0,0]
			fy = K[1,1]
			cx = K[0,2]
			cy = K[1,2]
			width = cam['resolution'][0]
			height = cam['resolution'][1]
			glPushMatrix()
			glMultMatrixf(camMatrix.T)
			glLineWidth(2)
			glColor3f(1,0,0)
			glBegin(GL_LINES)
			glVertex3f(0,0,0)
			glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz)
			glVertex3f(0,0,0)
			glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(0,0,0)
			glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(0,0,0)
			glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz)
			glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz)
			glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz)
			glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz)
			glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz)
			glEnd()
			glPopMatrix()


	def setFreeView(self):
		glTranslatef(0,0,self.zTrans)
		glRotatef( -self.yRot, 1.0, 0.0, 0.0)
		glRotatef( -self.xRot, 0.0, 1.0, 0.0)
		glRotatef( self.zRot, 0.0, 0.0, 1.0)
		glTranslatef( self.xTrans,  0.0, 0.0 )
		glTranslatef(  0.0, self.yTrans, 0.0)
		#print('zTrans {} xRot {} yRot {}  xTrans{}  yTrans'.format(self.zTrans,self.yRot,self.xRot,self.zRot,self.xTrans,self.yTrans))


	def mainloop(self):

		#layer0
		glBindFramebuffer(GL_FRAMEBUFFER,0)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		glEnable (GL_LINE_SMOOTH)
		glHint (GL_LINE_SMOOTH_HINT, GL_NICEST)


		self.lookAt()
		self.render_cameras()
		self.renderDomeFloor()
		cvt = self.meshs[self.frameId]['v'].astype(np.float32)
		cf =  self.meshs[self.frameId]['f'].astype(np.int)
		cvn =  self.meshs[self.frameId]['vn'].astype(np.float32)

		glPointSize(2.0)
		glBegin(GL_POINTS)
		#self.renderObj(cvt,cvn,cf,[0.85,0.85,0.85,1.0])

		glColor3f(0.0,0.0,0.0)
		for cver in cvt:
			glVertex3f(cver[0],cver[1],cver[2])

		glEnd()
		glEnable(GL_LIGHTING)

		cJtr25 = self.joints[self.frameId]
		cJtr25 = np.hstack((cJtr25,np.ones((25,1))))
		#print cJtr25
		self.renderJoints(cJtr25,[0.5,1.0,1.0,1])


		self.frameId = self.frameId+1
		if(self.frameId==self.frameNum):
			self.frameId = 0
		glutSwapBuffers()

	def reshapeCallback(self,width, height):
		self.width = width
		self.height = height
		glViewport(0, 0, self.width, self.height)

	def idlefuncCallback(self):
		glutPostRedisplay()

	def keyboardCallback(self,key, x, y):
		if key == chr(27) or key == "q":
			sys.exit()
		elif key == 's':
			#global width, height
			glReadBuffer(GL_FRONT)
			img = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, outputType=None)
			img = Image.frombytes("RGBA", (self.width,self.height), img)
			img = ImageOps.flip(img)
			img.save('temp.png', 'PNG')
		elif key == 'f':
			if self.viewMode=='free':
				self.viewMode = 'camera'
			else:
				self.viewMode ='free'
		elif key>='0' and key<='9':
			self.camId.popleft()
			self.camId.append(key)
		glutPostRedisplay()

	def mouseCallback(self,button, state, x, y):
		if (button==GLUT_LEFT_BUTTON):
			if (glutGetModifiers() == GLUT_ACTIVE_SHIFT):
				self.mouseAction = "TRANS"
			else:
				self.mouseAction = "MOVE_EYE"
		elif (button==GLUT_RIGHT_BUTTON):
			self.mouseAction = "ZOOM"
		self.xMousePosPrev = x
		self.yMousePosPrev = y

	def motionCallback(self,x, y):
		if (self.mouseAction=="MOVE_EYE"):
			self.xRot += x - self.xMousePosPrev
			self.yRot -= y - self.yMousePosPrev
		elif (self.mouseAction=="TRANS"):
			self.xTrans += x - self.xMousePosPrev
			self.yTrans += y - self.yMousePosPrev
		elif (self.mouseAction=="ZOOM"):
			self.zTrans -= y - self.yMousePosPrev
		else:
			print("unknown action\n", self.mouseAction)
		self.xMousePosPrev = x
		self.yMousePosPrev = y 
		glutPostRedisplay()


def parse_args():

	parser = argparse.ArgumentParser(description="")
	parser.add_argument(
		'--seqname',
		dest='seqname',
		default='161202_haggling1',
		type=str
	)
	parser.add_argument(
		'--type',
		dest='type',
		default='ADAM',
		type=str
	)
	parser.add_argument(
		'--skip',
		dest='skip',
		default = 0,
		type = int
	)
	parser.add_argument(
		'--frames',
		dest='frames',
		default=-1,
		type=int
	)
	return parser.parse_args()

	
if __name__ == '__main__':


	args = parse_args()

	calibFile = '/home/xiul/databag/dome_sptm/171204_pose6/calibration_171204_pose6.json'
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										
	glRender = domeViewer(1280,720)
	glRender.loadCalibs(calibFile)

	glRender.load_data()
	glRender.init_GLUT(sys.argv)