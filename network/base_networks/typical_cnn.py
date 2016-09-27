

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import Pool2DLayer as PoolLayerDNN
from lasagne.layers import Conv2DLayer as ConvLayer


from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear
import lasagne.layers
import settings
import os
import pickle
import os.path

import theano.tensor as T

class TypicalCNN(object):
    def __init__(self, input_var, out_size = 10):
        self.net = self.build_model(input_var)
        self.classes = range(out_size)
        self.mean_image = [0,0,0]
        self.name_features_layer = 'featsout'
        self.name_last_layer = 'probsout'

    def get_network(self):
        return self.net, self.classes, self.mean_image

    def build_model(self, input_var):
        net = {}
        
        net['conv1'] = lasagne.layers.Conv2DLayer(  input_var, 
                                                    num_filters=32, filter_size=(5, 5),
                                                    nonlinearity=lasagne.nonlinearities.rectify,
                                                    W=lasagne.init.GlorotUniform())
        net['mxpool1'] = lasagne.layers.MaxPool2DLayer( net['conv1'], pool_size=(2, 2))
        net['conv2'] = lasagne.layers.Conv2DLayer(  net['mxpool1'], 
                                                    num_filters=32, 
                                                    filter_size=(5, 5),
                                                    nonlinearity=lasagne.nonlinearities.rectify,
                                                    W=lasagne.init.GlorotUniform())
        net['mxpool2'] = lasagne.layers.MaxPool2DLayer( net['conv2'], pool_size=(2, 2))
        net['featsout'] = lasagne.layers.DenseLayer(
                                                    lasagne.layers.dropout(net['mxpool2'], p=.5),
                                                    num_units=256,
                                                    nonlinearity=lasagne.nonlinearities.rectify)
        net['probsout'] = lasagne.layers.DenseLayer(
                                                    lasagne.layers.dropout(net['featsout'], p=.5),
                                                    num_units=10,
                                                    nonlinearity=lasagne.nonlinearities.softmax)
        return net

    def get_name_features_layer(self):
        return self.name_features_layer

    def get_name_last_layer(self):
        return self.name_last_layer


class VideoTagsPredictorFramePerFrame(object):
    def __init__(self, input_var, frame_per_frame_network = TypicalCNN, out_size = 10):
        
        net_all = {}
        net_all['reshaped_input'] = Video2ImagePool(input_var)
        
        cnn = frame_per_frame_network(net_all['reshaped_input'], 10)
        net, classes, mean_image = cnn.get_network()
        output_layer = net[cnn.get_name_last_layer()]
        for key in net:
            net_all[key] = net[key]

        net_all['reshaped_output'] = ImagePoolToVideo(output_layer, video_batches, frames_per_video)    
        net_all['probsout'] = BatchAverageLayer(net_all['reshaped_output'])
        
        self.name_features_layer = 'reshaped_output'
        self.name_last_layer = 'probsout'

        self.mean_image = cnn.mean_image
        self.classes = cnn.classes
        self.net = net_all

    def get_name_features_layer(self):
        return self.name_features_layer

    def get_name_last_layer(self):
        return self.name_last_layer

    def get_network(self):
        return self.net, self.classes, self.mean_image


    

from datasets.dataset_creators.MNISTsequence import get_theano_dataset
from custom_layers.batch_average import ImagePoolToVideo, Video2ImagePool, BatchAverageLayer
import theano
import numpy
theano.config.optimizer = 'None'
theano.config.dnn.enabled = True



experiment_name = 'mnist_test'
videos_to_generate = {'Train':1000, 'Val':500, 'Test':500}
frames_per_video = 4
video_batches = 20
theano_dataset = get_theano_dataset(experiment_name, videos_to_generate, frames_per_video)



inp = lasagne.layers.InputLayer(shape=(video_batches, frames_per_video, 1, 28, 28),  input_var=theano_dataset.get_batch_data('Train'))
net = VideoTagsPredictorFramePerFrame(inp, frame_per_frame_network = TypicalCNN, out_size = 10)


net_layers, classes, mean_image = net.get_network()
out_feats_tensor = net_layers[net.get_name_last_layer()]
output = lasagne.layers.get_output(out_feats_tensor, deterministic = True)

inps = theano_dataset.get_input_tensors()
f = theano.function([inps['index'], inps['bsize']], output)

n_epochs = 10
from time import time
for epoch in range(n_epochs):
    print("Start epoch")    
    theano_dataset.shuffle_data('Train')
    num_batches = theano_dataset.get_num_batches('Train',video_batches)    
    for i_batch in range(num_batches):
        t1 = time()
        print("ei!")
        #X_train, y_train = theano_dataset.get_batch(i_batch, video_batches, 'Train')
        out = f(i_batch, video_batches)
        print(time()-t1)

    break

