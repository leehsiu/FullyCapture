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
import copy
import sklearn.preprocessing
from collections import deque


class GLUTviewer:
	def __init__(self,width,height):
		

		self.width = width
		self.height = height

		#render options
		self.zNear = 1.0
		self.zFar = 2000.0

		#ui_helper

		self.xMousePosPrev = 0
		self.yMousePosPrev = 0
		
		#main ui cam status
		self.xTrans = 0
		self.yTrans = 0
		self.zTrans = 450
		self.xRot = 60
		self.yRot = -40
		self.zRot = 0

		self.viewMode = 'free'
		self.frameId = 0
		self.camId = deque('00',maxlen=2)


		#params for joints render
		self.coco25_parents = [1,1,1,2,3,1,5,6,1,8,9,10,8,12,13,0,0,15,16,14,19,14,11,22,11]
		self.coco_inds = range(25)

		self.rhand_ids = range(25,45)
		self.rhand_parents = [4,1+24,2+24,3+24,4,5+24,6+24,7+24,4,9+24,10+24,11+24,4,13+24,14+24,15+24,4,17+24,18+24,19+24]
		
		self.lhand_ids = range(45,65)
		self.lhand_parents =[7,1+44,2+44,3+44,7,5+44,6+44,7+44,7,9+44,10+44,11+44,7,13+44,14+44,15+44,7,17+44,18+44,19+44]

		self.mouseAction = ""

		self.preload = False
		self.pause = False
		self.lighting = True
		self.frameNum = 0

		#defautl elements all one
		self.element = {'cameras':True,
					    'floor':True,
						'mesh':True,
						'joints':True,
						'points':True}
		
		self.element_func = {'mesh':None,
							 'joints':None,
							 'points':None}
		
	def initGLContent(self):
		#glClearColor(0.5, 0.5, 0.5, 0.0)
		glClearColor(1.0, 1.0, 1.0, 0.0)
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glShadeModel(GL_SMOOTH)
		glEnable(GL_LIGHTING)
		glEnable(GL_ALPHA_TEST)
		glEnable(GL_TEXTURE_2D)
		glEnable(GL_LIGHT0) 

		glEnable(GL_COLOR_MATERIAL)
		glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

		self.colorTex = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.colorTex)
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
	
		self.vc_buffers = glGenBuffers(1)
		self.vt_buffers = glGenBuffers(1)
		self.vn_buffers = glGenBuffers(1)
		self.f_buffers = glGenBuffers(1)

	def set_element_status(self,elem_name,elem_status):
		self.element[elem_name] = elem_status

	def start(self,argv):
		glutInitDisplayMode(GLUT_RGB|GLUT_DOUBLE|GLUT_DEPTH)
		glutInitWindowPosition(100,100)
		glutInitWindowSize(self.width,self.height)
		glutInit(argv)
		glutCreateWindow("GLUT viewer")
		self.initGLContent()
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


	def easyBnBCone(self,v1,v2,dim,color):
		glColor4f(color[0], color[1], color[2], color[3])
		v_mid  = (v2 + v1) / 2
		import math
		z = np.array([0.0, 0.0, 1.0])
		# the rotation axis is the cross product between Z and v2r
		ax0 = np.cross(z, v2 - v1)
		ax1 = np.cross(z, v1 - v2)
		l = math.sqrt(np.dot(v2 - v1, v2 - v1))
		# get the angle using a dot product
		angle0 = 180.0 / math.pi * math.acos(np.dot(z, v2 - v1)/l)
		angle1 = 180.0 / math.pi * math.acos(np.dot(z, v1 - v2)/l)
		glPushMatrix()
		glTranslatef(v_mid[0], v_mid[1], v_mid[2])

		#print "The cylinder between %s and %s has angle %f and axis %s\n" % (v1, v2, angle, ax)
		glRotatef(angle0, ax0[0], ax0[1], ax0[2])
		glutSolidCone(dim / 10.0, l/2, 20, 20)
		glPopMatrix()


		glPushMatrix()
		glTranslatef(v_mid[0], v_mid[1], v_mid[2])
		glRotatef(angle1,ax1[0],ax1[1],ax1[2])
		glutSolidCone(dim / 10.0, l/2, 20, 20)
		glPopMatrix()

	def easyCone(self,v1,v2,dim,color):
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
		glutSolidCone(dim / 10.0, l, 20, 20)
		glPopMatrix()

	

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

	
	def easySphere(self,v1,dim,color):
		glColor4f(color[0], color[1], color[2], color[3])

		glPushMatrix()
		glTranslatef(v1[0], v1[1], v1[2])
		
		glutSolidSphere(dim / 10.0, 20, 20)
		glPopMatrix()
	
			
	def set_load_func(self,element_name,load_func):
		if element_name not in self.element_func:
			raise ValueError('{} not in the supported elements, should be mesh,joints or points'.format(element_name))
		self.element_func[element_name] = load_func
	def set_frame_num(self,frame_num):
		self.frameNum = frame_num

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
			self.set_freeview()
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


	def bind_buffers(self,vt,vc,vn,f):
		num_faces = f.shape[0]

		vt_c = vt.flatten().tolist()
		vc_c = vc.flatten().tolist()
		vn_c = vn.flatten().tolist()
		f_c = f.flatten().astype(np.int).tolist()

		
		glBindBuffer(GL_ARRAY_BUFFER, self.vt_buffers)
		glBufferData(GL_ARRAY_BUFFER, len(vt_c)*sizeof(ctypes.c_float),
						(ctypes.c_float*len(vt_c))(*vt_c), GL_STATIC_DRAW)

		glBindBuffer(GL_ARRAY_BUFFER, self.vc_buffers)
		glBufferData(GL_ARRAY_BUFFER, len(vc_c)*sizeof(ctypes.c_float),
						(ctypes.c_float*len(vc_c))(*vc_c), GL_STATIC_DRAW)

		glBindBuffer(GL_ARRAY_BUFFER, self.vn_buffers)
		glBufferData(GL_ARRAY_BUFFER, len(vn_c)*sizeof(ctypes.c_float),
						(ctypes.c_float*len(vn_c))(*vn_c), GL_STATIC_DRAW)

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.f_buffers)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ctypes.c_int) * len(f_c),
						(ctypes.c_int * len(f_c))(*f_c), GL_STATIC_DRAW)
		
		return num_faces


	def render_buffers(self,num_faces,mode=GL_FILL):

		glEnable(GL_CULL_FACE)
		glPushMatrix()
		glLineWidth(.5)
		glEnableVertexAttribArray(0)
		glEnableVertexAttribArray(3)
		glBindBuffer(GL_ARRAY_BUFFER, self.vt_buffers)
		glVertexAttribPointer(
					0,                                # attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_FALSE,                         # normalized?
					0,  # / stride
					None                         # array buffer offset
				)
		glBindBuffer(GL_ARRAY_BUFFER, self.vc_buffers)
		glVertexAttribPointer(
					3,                                # 1 attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_FALSE,                         # normalized?
					0,  # / stride
					None                         # array buffer offset
				)

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
		
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.f_buffers)
		glPolygonMode(GL_FRONT_AND_BACK, mode)
		glDrawElements(GL_TRIANGLES, num_faces*3, GL_UNSIGNED_INT, None)

		glDisableVertexAttribArray(2)
		glDisableVertexAttribArray(0)
		glDisableVertexAttribArray(3)
		glPopMatrix()


	def render_joints(self,Jtr):

		for cid,pid in zip(self.coco_inds,self.coco25_parents):
			if cid==pid:
				continue	
			if Jtr[cid,3]>0.1 and Jtr[pid,3]>0.1:
				self.easyCone(Jtr[pid,0:3],Jtr[cid,0:3],15,[0.75,0.75,0.75,1.0])
		for cid,pid in zip(self.rhand_ids,self.rhand_parents):
			if cid==pid:
				continue	
			if Jtr[cid,3]>0.1 and Jtr[pid,3]>0.1:
				self.easyCone(Jtr[pid,0:3],Jtr[cid,0:3],5,[0.80,0.80,0.80,1.0])

		for cid,pid in zip(self.lhand_ids,self.lhand_parents):
			if cid==pid:
				continue
			if Jtr[cid,3]>0.1 and Jtr[pid,3]>0.1:
				self.easyCone(Jtr[pid,0:3],Jtr[cid,0:3],5,[0.80,0.80,0.80,1.0])
		
		


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

	def set_freeview(self):
		glTranslatef(0,0,self.zTrans)
		glRotatef( -self.yRot, 1.0, 0.0, 0.0)
		glRotatef( -self.xRot, 0.0, 1.0, 0.0)
		glRotatef( self.zRot, 0.0, 0.0, 1.0)
		glTranslatef( self.xTrans,  0.0, 0.0 )
		glTranslatef(  0.0, self.yTrans, 0.0)
		#print('zTrans {} xRot {} yRot {}  xTrans{}  yTrans'.format(self.zTrans,self.yRot,self.xRot,self.zRot,self.xTrans,self.yTrans))


	def mainloop(self):

		#layer0
		#time.sleep(0.1)
		glBindFramebuffer(GL_FRAMEBUFFER,0)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		glEnable (GL_LINE_SMOOTH)
		glHint (GL_LINE_SMOOTH_HINT, GL_NICEST)

		self.lookAt()
		if(self.element['cameras']):
			self.render_cameras()
		if(self.element['floor']):
			self.renderDomeFloor()

		if(self.element['mesh']):
			#glEnable(GL_LIGHTING)
			if self.lighting:
				glEnable(GL_LIGHTING)
			else:
				glDisable(GL_LIGHTING)
			if self.element_func['mesh'] is None:
				raise NotImplementedError('load function for mesh has not been specified')
				#not preload.
			vts,vcs,vns,vf = self.element_func['mesh'](self.frameId)
			num_faces = self.bind_buffers(vts,vcs,vns,vf)
			self.render_buffers(num_faces)
		
		if(self.element['joints']):
			glEnable(GL_LIGHTING)
			if self.element_func['joints'] is None:
				raise NotImplementedError('load function for joints has not been specified')
			joints_total = self.element_func['joints'](self.frameId)
			for joints in joints_total:
				self.render_joints(joints)
		if not self.pause:
			#next frameId
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
		elif key == 'p':
			if self.pause:
				self.pause = False
			else:
				self.pause = True
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