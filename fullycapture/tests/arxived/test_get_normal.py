import glob
import os.path
from totaldensify.optimizer.std_capture import StdCapture
import torch
import sys
import json
import numpy as np
import numpy.linalg as nlg
import cv2
import cPickle as pickle
import time
import totaldensify.data.dataIO as dataIO
import totaldensify.utils.config as cfg_utils
from totaldensify.utils.config import cfg

if __name__=='__main__':

    cfg_utils.merge_cfg_from_file('../../config/TotalCapture.yaml')
    cfg_utils.assert_and_infer_cfg()

    capUtils = StdCapture(cfg.CAPTURE.MODEL_FEMALE,cfg.CAPTURE.MODEL_MALE,reg_type='total')
    