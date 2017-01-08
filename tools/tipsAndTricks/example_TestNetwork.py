from network.base_networks.googlenet import GooglenetNetwork
import lasagne
from lasagne.layers import set_all_param_values
import pickle
import numpy as np
import cv2
from custom_layers.batch_average import BatchAverageLayer

def loadImage(imagepath, mean_image):
    img = cv2.resize(cv2.imread(imagepath), (224, 224))
    mean_pixel = mean_image # [103.939, 116.779, 123.68]
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)
    return img

print("starting")
batch_size = 12
frames_per_video = 6

print("defining and loading network")
net = GooglenetNetwork(batch_size)
net, classes, mean_image = net.get_network()

print("adding my own layer")
net['aggregate_per_video'] = BatchAverageLayer(net['loss3/classifier'], frames_per_video = frames_per_video)

output_layer = net['aggregate_per_video']

print("preparing input...")
cat_image = loadImage('../../Desktop/cat.jpg', mean_image)
bed_image = loadImage('../../Desktop/bed.jpg', mean_image)

data_to_feed = np.vstack([cat_image]*frames_per_video + [bed_image]*frames_per_video)

print("feeding input...")

classification_result = lasagne.layers.get_output(output_layer, data_to_feed)

print(data_to_feed.shape)
print(classification_result.eval().shape)

print(map(lambda x: classes[x], classification_result.eval().argmax(axis=1)))
