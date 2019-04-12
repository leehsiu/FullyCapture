import torch
import torch.nn
import numpy as np
import cv2

img = cv2.imread('./joints.png')

#grid sample.

#N,V,1,2 the input postions.. get from projection. So its V, projected by multiple views and parameters. 
#V projected by theta,betas, can get (x,y) as input. then extract features from image. using the pre-computed visibility map. on each 
#view and frame as weight. compare the color difference.
