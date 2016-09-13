from network.VGG16 import VGG16Network
import lasagne
from lasagne.layers import set_all_param_values
import pickle
import numpy as np
import cv2
from custom_layers import BatchAverageLayer

def loadImage(imagepath):
    img = cv2.resize(cv2.imread(imagepath), (224, 224))
    mean_pixel = MEAN_IMAGE # [103.939, 116.779, 123.68]
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)
    return img


frames_per_video = 12

net = VGG16Network()
net, classes, mean_image = net.get_network()
net['aggregate_per_video'] = BatchAverageLayer(net['fc8'], frames_per_video = frames_per_video)

cat_image = loadImage('../../Desktop/cat.jpg')
bed_image = loadImage('../../Desktop/bed.jpg')

data_to_feed = np.vstack([cat_image]*frames_per_video + [bed_image]*frames_per_video)

classification_result = lasagne.layers.get_output(output_layer, data_to_feed)

print(data_to_feed.shape)
print(classification_result.shape)

