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

imgIUV0 = cv2.imread('/home/xiul/databag/dome_sptm/170407_haggling_b3/sample/results/iuv2.png')
imgIUV1 = cv2.imread('/home/xiul/databag/dome_sptm/170407_haggling_b3/sample/results/iuv1.png')

I_to_mask = [0,1,1,2,3,4,5,6,7,6,7,8,9,8,9,10,11,10,11,12,13,12,13,14,14]

imgMask = np.zeros((720,1280),dtype=np.float32)
for ii in range(1,25):
    uv0 = imgIUV0[:,:,0]
    imgMask[uv0==ii] = I_to_mask[ii]
ax0.clear()
ax0.imshow(imgRGB1[:,:,::-1])
ax0.imshow(imgMask,alpha=0.5)

ax0.set_xticks(())
ax0.set_yticks(())
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

# plt.show()
# plt.draw()
# plt.pause(0.01)
# fig.set_size_inches(8.5, 10.5)
# fig.savefig('/home/xiul/databag/dome_sptm/170407_haggling_b3/sample/overlay1.png', bbox_inches = 'tight',pad_inches = 0)
