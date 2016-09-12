from network.base_VGG_16_layer import build_model
import lasagne
from lasagne.layers import set_all_param_values
import pickle
import numpy as np
import cv2


def loadImage(imagepath):
    img = cv2.resize(cv2.imread(imagepath), (224, 224))
    mean_pixel = MEAN_IMAGE # [103.939, 116.779, 123.68]
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)
    return img

net = build_model()

with open('data/network/vgg16.pkl', 'rb') as f:
    params = pickle.load(f)


output_layer = net['fc8']

import pickle
CLASSES = params['synset words']
MEAN_IMAGE = params['mean value']
lasagne.layers.set_all_param_values(output_layer, params['param values'])


classification_result = lasagne.layers.get_output(output_layer, loadImage('../../Desktop/cat.jpg')).eval()
print(classification_result)

classification_result = lasagne.layers.get_output(output_layer, loadImage('../../Desktop/bed.jpg')).eval()
print(classification_result)

classification_result = lasagne.layers.get_output(output_layer, loadImage('../../Desktop/futbol.jpg')).eval()
print(classification_result)


