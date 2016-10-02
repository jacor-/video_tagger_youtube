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




from datasets.dataset_creators.MNISTsequence import get_theano_dataset
from custom_layers.batch_average import ImagePoolToVideo, Video2ImagePool, VideoSummarySumSigmoidLayer
import theano
import numpy
theano.config.optimizer = 'None'
theano.config.dnn.enabled = 'True'





def buildNetwork_tensor(inp_X, network_name, per_frame_network, out_size, video_batches, frames_per_video):
    # We build the network architecture. The final part will define how the network is trained, so we keep it
    # within the if statement.

    video_unpooled = Video2ImagePool(inp_X)
    cnnet_out =      per_frame_network(video_unpooled, out_size)
    video_pooled =   ImagePoolToVideo(cnnet_out, video_batches, frames_per_video)
    if network_name == 'mlp_simple_aggregation':
        # We simply aggregate all the frames summing them and aplying a sigmoid layer to that sum
        final_classification = VideoSummarySumSigmoidLayer(video_pooled)
    else:
        raise "%s is not implemented yet!" % network_name

    # We get the tensors for the outputs of the net: frames prediction and aggregation prediction
    network = final_classification
    output_features_tensor = lasagne.layers.get_output(video_pooled, deterministic = True)
    output_classification_tensor = lasagne.layers.get_output(final_classification, deterministic = True)
    return output_features_tensor, output_classification_tensor, network





class OutputNetworkManager(object):
    def __init__(self, videos_per_batch, dataset, input_tensors):
        self.videos_per_batch = videos_per_batch
        self.dataset = dataset
        self.input_tensors = input_tensors

        self.f_get_set_outputs = self._prepare_get_output_functions()


    def _prepare_get_output_functions(self):
        out = {}
        for set_ in ('Train','Test','Val'):
            X_batch, y_batch_video, y_batch_frame = self.dataset.get_tensor_batch_data(set_), self.dataset.get_tensor_batch_video_labels(set_), self.dataset.get_tensor_batch_frame_labels(set_)
            f_video_output = theano.function(
                                            [input_tensors['index'], input_tensors['bsize']],
                                            [out_frame_tensor, input_tensors['y_frames'], out_video_tensor, input_tensors['y_videos']],
                                            givens = {input_tensors['inp_data']:X_batch,
                                                      input_tensors['y_videos']:y_batch_video,
                                                      input_tensors['y_frames']:y_batch_frame
                                                      }
                                           )
            out[set_] = f_video_output
        return out

    def getOutputForSet(self, set_):
        out_frame, y_frames, out_video, y_video = [], [], [], []
        for i_batch in range(self.dataset.get_num_batches(set_, self.videos_per_batch)):
            out_frame_, y_frames_, out_video_, y_video_ = self.f_get_set_outputs[set_](i_batch, self.videos_per_batch)
            out_frame.append(out_frame_)
            y_frames.append(y_frames_)
            out_video.append(out_video_)
            y_video.append(y_video_)
        return np.vstack(out_frame), np.vstack(y_frames), np.vstack(out_video), np.vstack(y_video)





frames_per_video = 4
video_batches = 20
out_size = 10

## Dataset definition
experiment_name = 'mnist_test'
videos_to_generate = {'Train':10000, 'Val':1000, 'Test':3000}

theano_dataset = get_theano_dataset(experiment_name, videos_to_generate, frames_per_video)
inp_index_batch = theano_dataset.get_input_tensors()
inp_X =        lasagne.layers.InputLayer(shape=(video_batches, frames_per_video, 1, 28, 28))
inp_Y_frames = lasagne.layers.InputLayer(shape=(video_batches, frames_per_video, out_size))
inp_Y_videos = lasagne.layers.InputLayer(shape=(video_batches, out_size))
input_tensors = { 'index' : inp_index_batch['index'], 'bsize': inp_index_batch['bsize'], 'inp_data' : inp_X.input_var, 'y_videos' : inp_Y_videos.input_var, 'y_frames' : inp_Y_frames.input_var }


out_frame_tensor, out_video_tensor, network = buildNetwork_tensor(inp_X, 'mlp_simple_aggregation', TypicalCNN, out_size, video_batches, frames_per_video)

netm = OutputNetworkManager(video_batches, theano_dataset, input_tensors)


set_ = 'Train'



video_reference = input_tensors['y_videos']
video_prediction = out_video_tensor
#video_reference = input_tensors['y_frames']
#video_prediction = out_frame_tensor


loss = lasagne.objectives.categorical_crossentropy(video_prediction, video_reference)
loss = -video_reference*T.log(video_prediction) - (1-video_reference)*T.log(1-video_prediction)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
print(params)

updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
#updates = lasagne.updates.sgd(loss, params, learning_rate=0.05)

X_batch, y_batch_video, y_batch_frame = theano_dataset.get_tensor_batch_data('Train'), theano_dataset.get_tensor_batch_video_labels('Train'), theano_dataset.get_tensor_batch_frame_labels('Train')
train_fn = theano.function([input_tensors['index'], input_tensors['bsize']],
                            loss,
                            givens = {  input_tensors['inp_data']:X_batch,
                                        input_tensors['y_videos']:y_batch_video,
                                        input_tensors['y_frames']:y_batch_frame
                                      },
                            updates = updates)


check_train_fn = theano.function([input_tensors['index'], input_tensors['bsize']],
                                    [out_frame_tensor, out_video_tensor, y_batch_video, y_batch_frame, loss],
                                    givens = {  input_tensors['inp_data']:X_batch,
                                                input_tensors['y_videos']:y_batch_video,
                                                input_tensors['y_frames']:y_batch_frame
                                              },
                                    updates = updates)

out_f_ini, out_v_ini, y_v_ini, y_f_ini, loss_ini = check_train_fn(0, video_batches)
#print((-y_f_ini*np.log(out_f_ini)-(1-y_f_ini)*np.log(1-out_f_ini)).mean())
print((-y_v_ini*np.log(out_v_ini)-(1-y_v_ini)*np.log(1-out_v_ini)).mean())
print(loss_ini)

losses = []
for n_epoch in range(25):
    t1 = time.time()
    print("Epoch : %d" % n_epoch)
    epoch_loss = []
    for i_batch in range(theano_dataset.get_num_batches('Train', video_batches)):
        loss = train_fn(i_batch, video_batches)
        epoch_loss.append(loss)
    losses.append(np.mean(epoch_loss))
    print("Loss : %0.5f" % losses[-1])
    print(time.time() - t1)
    theano_dataset.shuffle_data('Train')


#out_f_end, out_v_end, y_v_end, y_f_end, loss_end = test_fn(0, video_batches)
def printResults(set_):
    out_frame, y_frames, out_video, y_video = netm.getOutputForSet(set_)

    vid_general_accuracy = float((y_video == (out_video > 0.5)).sum()) / np.multiply(*out_video.shape)
    frame_accuracy = float((y_frames.argmax(axis=2) == out_frame.argmax(axis=2)).sum()) / np.multiply(*out_frame.shape[:2])

    print("Results for set " + set_)
    print(" - Frame accuracy: " + str(frame_accuracy))
    print(" - Video accuracy: " + str(vid_general_accuracy))

printResults('Train')
printResults('Test')
printResults('Val')

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