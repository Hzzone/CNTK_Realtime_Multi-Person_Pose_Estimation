from __future__ import print_function
import os
import numpy as np
import cntk as C
from cntk import load_model, combine
import cntk.io.transforms as xforms
from cntk.logging import graph
from cntk.logging.graph import get_node_outputs
import cv2 as cv
import scipy
import PIL.Image
import math
import sys
import time
import util
import copy
import matplotlib
import pylab as plt



# define location of model and data and check existence
model_file = os.path.join("..", "model", "pose_net.cntkmodel")

use_gpu = False

# scale_search = [0.5, 1, 1.5, 2]
scale_search = [0.5]
boxsize = 368
GPUdeviceNumber = 0
stride = 8
padValue = 128
thre1 = 0.1
thre2 = 0.05

test_image = os.path.join('.', 'sample.jpg')

def eval_single_image(loaded_model, image):
	# load and format image (resize, RGB -> BGR, CHW -> HWC)
	print(image.shape)
	bgr_image = np.asarray(image, dtype=np.float32)[..., [2, 1, 0]]
	print(bgr_image.shape)
	chw_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2)) / 256 - 0.5
	mchw_format = np.reshape(chw_format, (1, chw_format.shape[0], chw_format.shape[1], chw_format.shape[2]))
	print(mchw_format.shape)
	# compute model output
	print(type(loaded_model.arguments[0]))
	print(dir(loaded_model.arguments[0]))
	print(loaded_model.arguments[0].shape)

	x = C.input_variable(shape=(3, C.FreeDimension, C.FreeDimension), name="data")
	print(type(x))
	arguments = {x: [mchw_format]}
	output = loaded_model.eval(arguments)
	# my_model = loaded_model(x)

if __name__ == '__main__':
	oriImg = cv.imread(test_image)  # B,G,R order

	multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]

	for m in range(len(multiplier)):
		scale = multiplier[m]
		imageToTest = cv.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
		imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
		print(imageToTest.shape)

	trained_model = C.load_model(model_file)
	eval_single_image(trained_model, imageToTest_padded)
