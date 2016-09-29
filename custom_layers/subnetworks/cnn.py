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

###
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import Pool2DLayer as PoolLayerDNN
from lasagne.layers import Conv2DLayer as ConvLayer
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
        self.net = self.build_model(incoming)

    def get_output_for(self, input, **kwargs):
        return lasagne.layers.get_output(self.net[self.name_last_layer], inputs=input)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.out_size)

    def get_useful_params(self):
        return {'mean_image': self.mean_image}

    def build_model(self, incoming):
        net = {}

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

        ### The last layer is a tanh to be sure it is multiclass (for mnist it does not matter, but we want do so something scalable)
        net['probsout'] = lasagne.layers.DenseLayer(
                                                    lasagne.layers.dropout(net['featsout'], p=.5),
                                                    num_units=self.out_size,
                                                    nonlinearity=lasagne.nonlinearities.tanh)
        return net




from datasets.dataset_creators.MNISTsequence import get_theano_dataset
from custom_layers.batch_average import ImagePoolToVideo, Video2ImagePool, VideoSummarySumSigmoidLayer
import theano
import numpy
theano.config.optimizer = 'None'
theano.config.dnn.enabled = 'True'





def buildNetwork_tensor(inp_X, network_name, out_size, video_batches, frames_per_video):
    # We build the network architecture. The final part will define how the network is trained, so we keep
    # within the if statement.
    video_unpooled = Video2ImagePool(inp_X)
    cnnet_out =      TypicalCNN(video_unpooled, out_size)
    video_pooled =   ImagePoolToVideo(cnnet_out, video_batches, frames_per_video)
    if network_name == 'mlp_simple_aggregation':
        # We simply aggregate all the frames summing them and aplying a sigmoid layer to that sum
        final_classification = VideoSummarySumSigmoidLayer(video_pooled)
    else:
        raise "%s is not implemented yet!" % network_name

    # We get the tensors for the outputs of the net: frames prediction and aggregation prediction
    output_features_tensor = lasagne.layers.get_output(video_pooled, deterministic = True)
    output_classification_tensor = lasagne.layers.get_output(final_classification, deterministic = True)
    return output_features_tensor, output_classification_tensor

def get_loss_frames_tensor(inp_Y_frames, output_features_tensor):
    cost_functions = {}
    cost_functions['cross_entropy'] = lasagne.objectives.binary_crossentropy(inp_Y_frames, output_features_tensor).sum(axis=1).mean()
    cost_functions['strict_accuracy'] = (inp_Y_video == output_video_tensor >= 0.5).sum(axis = 1).sum() / (inp_Y_frames[0] * inp_Y_frames[1])

def get_loss_frames_video(inp_Y_video, output_video_tensor):
    cost_functions = {}
    eqs = T.cast(T.eq(inp_Y_videos, output_classification_tensor >= 0.5),'floatX').sum(axis=1)
    cost_functions['strict_accuracy'] = T.cast(T.eq(eqs,inp_Y_video.shape[1]), 'float32').sum() / inp_Y_video.shape[0]
    cost_functions['cross_entropy'] = lasagne.objectives.binary_crossentropy(inp_Y_video, output_video_tensor).sum(axis=1).mean()

def getLossFunction(theano_dataset, set_):
    X_batch, y_batch = theano_dataset.get_batch_data(set_), theano_dataset.get_batch_labels(set_)
    f_video_loss = theano.function(
                                    [inps['index'], inps['bsize']],
                                    [video_losses['strict_accuracy'], video_losses['cross_entropy']],
                                    givens = {inp_X.input_var:X_batch,
                                              inp_Y_video.input_var:Y_batch
                                              }
                                   )
    return f_video_loss



frames_per_video = 4
video_batches = 20
out_size = 10

## Dataset definition
experiment_name = 'mnist_test'
videos_to_generate = {'Train':1000, 'Val':500, 'Test':500}

theano_dataset = get_theano_dataset(experiment_name, videos_to_generate, frames_per_video)
inps = theano_dataset.get_input_tensors()

inp_X =        lasagne.layers.InputLayer(shape=(video_batches, frames_per_video, 1, 28, 28))
inp_Y_frames = lasagne.layers.InputLayer(shape=(video_batches, frames_per_video, out_size))
inp_Y_videos = lasagne.layers.InputLayer(shape=(video_batches, out_size))

out_frame_tensor, out_video_tensor = buildNetwork_tensor(inp_X, 'mlp_simple_aggregation', out_size, video_batches, frames_per_video)
#video_losses = get_loss_frames_video(lasagne.layers.get_output(inp_Y_videos)[0], output_classification_tensor)



set_ = 'Train'
X_batch, y_batch_video, y_batch_frame = theano_dataset.get_tensor_batch_data(set_), theano_dataset.get_tensor_batch_video_labels(set_), theano_dataset.get_tensor_batch_frame_labels(set_)
f_video_loss = theano.function(
                                [inps['index'], inps['bsize']],
                                [inp_Y_videos.input_var, inp_X.input_var],
                                givens = {inp_X.input_var:X_batch,
                                          inp_Y_videos.input_var:y_batch_video
                                          }
                               )





'''

n_epochs = 10
from time import time
for epoch in range(n_epochs):

    t1 = time()

    print("Start epoch")
    theano_dataset.shuffle_data('Train')
    num_batches = theano_dataset.get_num_batches('Train',video_batches)
    for i_batch in range(num_batches):
        #X_train, y_train = theano_dataset.get_batch(i_batch, video_batches, 'Train')
        out = f(i_batch, video_batches)

    print(time()-t1)
'''