import lasagne
import lasagne.layers
import numpy as np
import theano
import theano.tensor as T
import lasagne.nonlinearities

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
import time
###
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
#from lasagne.layers import Pool2DLayer as PoolLayerDNN
#from lasagne.layers import Conv2DLayer as ConvLayer
###

from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear
import lasagne.layers
import settings
import os
import pickle
import os.path

import theano.tensor as T

from custom_layers.batch_average import VideoSummarySumSigmoidLayer, ImagePoolToVideo




class TypicalCNN(lasagne.layers.Layer):
    def __init__(self, incoming, out_size, **kwargs):
        super(TypicalCNN, self).__init__(incoming, **kwargs)
        self.name_last_layer = 'probsout'
        self.out_size = out_size
        self.mean_image = [0,0,0]
        self.net, inner_params = self.build_model(incoming)

        for i, param in enumerate(inner_params):
            self.add_param(param, param.shape.eval(), name=param.name + "_" + str(i))


    def get_output_for(self, input, **kwargs):
        return lasagne.layers.get_output(self.net[self.name_last_layer], inputs=input)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.out_size)

    def get_useful_params(self):
        return {'mean_image': self.mean_image}

    def build_model(self, incoming):
        net = {}
        params = {}
        net['conv1'] = lasagne.layers.Conv2DLayer(  incoming,
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

        ### The last layer is a sigmoid to be sure it is multiclass (for mnist it does not matter, but we want do so something scalable)
        ## In the original version this was a softmax, which is not useful for multiclass
        net['probsout'] = lasagne.layers.DenseLayer(
                                                    lasagne.layers.dropout(net['featsout'], p=.5),
                                                    num_units=self.out_size,
                                                    nonlinearity=lasagne.nonlinearities.sigmoid)


        params = lasagne.layers.get_all_params(net['probsout'], trainable=True)
        return net, params

