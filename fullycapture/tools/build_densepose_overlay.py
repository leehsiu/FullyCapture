import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pycocotools.coco import COCO

fig,ax0 = plt.subplots(1,1)

img0_path  = '/home/xiul/databag/dome_sptm/170407_haggling_b3/sample/00_11_00003524.jpg'
img1_path = '/home/xiul/databag/dome_sptm/170407_haggling_b3/sample/00_18_00003524.jpg'

imgRGB0 = cv2.imread(img0_path)
imgRGB1 = cv2.imread(img1_path)

imgIUV0 = cv2.imread('/home/xiul/databag/dome_sptm/170407_haggling_b3/sample/iuv1.png')
imgIUV1 = cv2.imread('/home/xiul/databag/dome_sptm/170407_haggling_b3/sample/iuv2.png')

ax0.clear()
ax0.imshow(imgRGB1[:,:,::-1])
ax0.contour(imgIUV1[:,:,1]/256.0,8)
ax0.contour(imgIUV1[:,:,2]/256.0,8)
ax0.set_xticks(())
ax0.set_yticks(())
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.draw()
plt.pause(0.01)
fig.set_size_inches(8.5, 10.5)
fig.savefig('/home/xiul/databag/dome_sptm/170407_haggling_b3/sample/overlay1.png', bbox_inches = 'tight',pad_inches = 0)

#raw_input()
