# -*- coding: utf-8 -*-
from PIL import Image
import cntk as C
import numpy as np
from cntk import load_model, combine, CloneMethod
from cntk.layers import placeholder
from cntk.logging.graph import find_by_name
import cv2

# rgb_image = np.asarray(Image.open("3.png"), dtype=np.float32)
# rgb_image = np.asarray(Image.open("3.png"), dtype=np.float32) - 128
# print rgb_image.shape
# bgr_image = rgb_image[..., [2, 1, 0]]
# print bgr_image.shape
# pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
# print pic.shape

im = np.ascontiguousarray(np.transpose(cv2.imread("3.png"), (2, 0, 1)).astype(np.float32) - 128)
# print im.shape
# if (im == pic).all():
#     print u"相等"


base_model = load_model("model/pose_net.cntkmodel")
data = C.input_variable(shape=(3, C.FreeDimension, C.FreeDimension), name="data")
def clone_model(base_model, from_node_names, to_node_names, clone_method):
    from_nodes = [find_by_name(base_model, node_name) for node_name in from_node_names]
    if None in from_nodes:
        print("Error: could not find all specified 'from_nodes' in clone. Looking for {}, found {}"
              .format(from_node_names, from_nodes))
    to_nodes = [find_by_name(base_model, node_name) for node_name in to_node_names]
    if None in to_nodes:
        print("Error: could not find all specified 'to_nodes' in clone. Looking for {}, found {}"
              .format(to_node_names, to_nodes))
    input_placeholders = dict(zip(from_nodes, [placeholder() for x in from_nodes]))
    cloned_net = combine(to_nodes).clone(clone_method, input_placeholders)
    return cloned_net

predictor = clone_model(base_model, ['data'], ["Mconv7_stage6_L1", "Mconv7_stage6_L2"], CloneMethod.freeze)
pred_net = predictor(data)
print type(pred_net)
Mconv7_stage6_L1 = pred_net.outputs[0]
Mconv7_stage6_L2 = pred_net.outputs[1]
output = pred_net.eval({pred_net.arguments[0]: [im]})
print output[Mconv7_stage6_L1].shape
print output[Mconv7_stage6_L2]
