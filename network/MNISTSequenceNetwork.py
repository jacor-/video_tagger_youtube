import lasagne
import lasagne.layers
from numpy import random
import numpy as np
import theano
import theano.tensor as T
import lasagne.nonlinearities
import time
###
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
#from lasagne.layers import Pool2DLayer as PoolLayerDNN
#from lasagne.layers import Conv2DLayer as ConvLayer
###
import settings
import os
import pickle
import os.path
from datasets.dataset_creators.MNISTsequence import get_theano_dataset
from custom_layers.batch_average import ImagePoolToVideo, Video2ImagePool, VideoSummarySumSigmoidLayer, VideoSummaryMaxLayer, VideoSummarySumTanhLayer, VideoSummaryPoissonBernoulli
from custom_layers.subnetworks.cnn import TypicalCNN

#theano.config.optimizer = 'None'
#theano.config.dnn.enabled = 'True'

class OutputNetworkManager(object):
    def __init__(self, videos_per_batch, dataset, input_tensors, output_tensors):
        self.videos_per_batch = videos_per_batch
        self.dataset = dataset
        self.input_tensors = input_tensors
        self.out_frame_tensor, self.out_video_tensor = output_tensors

        self.f_get_set_outputs = self._prepare_get_output_functions()


    def _prepare_get_output_functions(self):
        out = {}
        for set_ in ('Train','Test','Val'):
            X_batch, y_batch_video, y_batch_frame = self.dataset.get_tensor_batch_data(set_), self.dataset.get_tensor_batch_video_labels(set_), self.dataset.get_tensor_batch_frame_labels(set_)
            f_video_output = theano.function(
                                            [self.input_tensors['index'], self.input_tensors['bsize']],
                                            [self.out_frame_tensor, self.input_tensors['y_frames'].input_var, self.out_video_tensor, self.input_tensors['y_videos'].input_var],
                                            givens = {self.input_tensors['inp_data'].input_var:X_batch,
                                                      self.input_tensors['y_videos'].input_var:y_batch_video,
                                                      self.input_tensors['y_frames'].input_var:y_batch_frame
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

class Network(object):

    def __init__(self, video_batches, frames_per_video, out_size, aggregation, base_net, theano_dataset):
        self.video_batches, self.frames_per_video, self.out_size = video_batches, frames_per_video, out_size
        self.aggregation = aggregation
        self.base_net = base_net

        self.__prepare_input_dataset(theano_dataset)
        self.__prepareNetwork(theano_dataset)
        self.__prepare_loss()
        self.__prepare_train_function(theano_dataset)


    def __buildNetwork_tensor(self):
        # We build the network architecture. The final part will define how the network is trained, so we keep it
        # within the if statement.

        video_unpooled = Video2ImagePool(self.input_tensors['inp_data'])
        cnnet_out =      self.base_net(video_unpooled, self.out_size)
        video_pooled =   ImagePoolToVideo(cnnet_out, self.video_batches, self.frames_per_video)
        if self.aggregation == 'tanh_aggregation':
            # We simply aggregate all the frames summing them and aplying a sigmoid layer to that sum
            final_classification = VideoSummarySumTanhLayer(video_pooled)
        elif self.aggregation == 'sigmoid_aggregation':
            # We simply aggregate all the frames summing them and aplying a sigmoid layer to that sum
            final_classification = VideoSummarySumSigmoidLayer(video_pooled)
        elif self.aggregation == 'max_aggregation':
            # We simply aggregate all the frames summing them and aplying a sigmoid layer to that sum
            final_classification = VideoSummaryMaxLayer(video_pooled)
        elif self.aggregation == 'poissonbernoulli_aggregation':
            final_classification = VideoSummaryPoissonBernoulli(video_pooled)
        else:
            raise "%s is not implemented yet!" % self.aggregation

        # We get the tensors for the outputs of the net: frames prediction and aggregation prediction
        network = final_classification
        output_features_tensor = lasagne.layers.get_output(video_pooled, deterministic = True)
        output_classification_tensor = lasagne.layers.get_output(final_classification, deterministic = True)
        return output_features_tensor, output_classification_tensor, network

    def __prepareNetwork(self, theano_dataset):
        self.out_frame_tensor, self.out_video_tensor, self.network = self.__buildNetwork_tensor()
        self.netm = OutputNetworkManager(self.video_batches, theano_dataset, self.input_tensors, (self.out_frame_tensor, self.out_video_tensor))
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)

    def __prepare_input_dataset(self, theano_dataset):
        inp_index_batch = theano_dataset.get_input_tensors()
        inp_X =        lasagne.layers.InputLayer(shape=(self.video_batches, self.frames_per_video, 1, 28, 28))
        inp_Y_frames = lasagne.layers.InputLayer(shape=(self.video_batches, self.frames_per_video, self.out_size))
        inp_Y_videos = lasagne.layers.InputLayer(shape=(self.video_batches, self.out_size))
        input_tensors = { 'index' : inp_index_batch['index'], 'bsize': inp_index_batch['bsize'], 'inp_data' : inp_X, 'y_videos' : inp_Y_videos, 'y_frames' : inp_Y_frames }
        self.input_tensors = input_tensors


    def __prepare_loss(self):

        video_reference = self.input_tensors['y_videos'].input_var
        video_prediction = self.out_video_tensor

        loss = lasagne.objectives.categorical_crossentropy(video_prediction, video_reference)
        loss = -video_reference*T.log(video_prediction) - (1-video_reference)*T.log(1-video_prediction)
        loss = loss.mean()

        self.loss = loss

    def __prepare_train_function(self, theano_dataset):
        updates = lasagne.updates.nesterov_momentum(self.loss, self.params, learning_rate=0.01, momentum=0.9)

        X_batch, y_batch_video, y_batch_frame = theano_dataset.get_tensor_batch_data('Train'), theano_dataset.get_tensor_batch_video_labels('Train'), theano_dataset.get_tensor_batch_frame_labels('Train')
        self.train_fn = theano.function([self.input_tensors['index'], self.input_tensors['bsize']],
                                        self.loss,
                                        givens = {  self.input_tensors['inp_data'].input_var:X_batch,
                                                    self.input_tensors['y_videos'].input_var:y_batch_video,
                                                    self.input_tensors['y_frames'].input_var:y_batch_frame
                                                  },
                                        updates = updates)
    def train(self, nepochs, theano_dataset, collect_metrics_for = ['Val']):

        val_metrics = {'video_acc':[], 'frame_acc':[], 'loss':[]}
        train_metrics = {'video_acc':[], 'frame_acc':[], 'loss':[]}
        test_metrics = {'video_acc':[], 'frame_acc':[], 'loss':[]}
        metrics = {'Train': train_metrics, 'Val': val_metrics, 'Test': test_metrics}

        val_video_accuracy = []
        val_frame_accuracy = []

        losses = []
        for n_epoch in range(nepochs):
            t1 = time.time()
            print("Epoch : %d" % n_epoch)
            epoch_loss = []
            for i_batch in range(theano_dataset.get_num_batches('Train', video_batches)):
                loss = self.train_fn(i_batch, video_batches)
                epoch_loss.append(loss)
            losses.append(np.mean(epoch_loss))

            for set_ in collect_metrics_for:
                out_frame, y_frames, out_video, y_video = self.netm.getOutputForSet(set_)
                loss, vac, fac = self.getResults(out_frame, y_frames, out_video, y_video)
                metrics[set_]['video_acc'].append(vac)
                metrics[set_]['frame_acc'].append(fac)
                metrics[set_]['loss'].append(loss)

            print("Loss : %0.5f" % losses[-1])
            print("Time : %0.4f" % (time.time() - t1))
            theano_dataset.shuffle_data('Train')

        return metrics

    def getResults(self, out_frame, y_frames, out_video, y_video):
        vid_general_accuracy = float((y_video == (out_video > 0.5)).sum()) / np.multiply(*out_video.shape)
        frame_accuracy = float((y_frames.argmax(axis=2) == out_frame.argmax(axis=2)).sum()) / np.multiply(*out_frame.shape[:2])
        losses = -(y_video * np.log(out_video) -(1-y_video) * np.log(1-out_video) ).sum() / np.multiply(*out_video.shape)
        return losses, vid_general_accuracy, frame_accuracy

    #out_f_end, out_v_end, y_v_end, y_f_end, loss_end = test_fn(0, video_batches)
    def printResults(self, set_):
        out_frame, y_frames, out_video, y_video = self.netm.getOutputForSet(set_)
        losses, vid_general_accuracy, frame_accuracy = self.getResults(out_frame, y_frames, out_video, y_video)
        print("Results for set " + set_)
        print(" - Frame accuracy: " + str(frame_accuracy))
        print(" - Video accuracy: " + str(vid_general_accuracy))
        print(" - Loss: " + str(losses))





frames_per_video = 3
video_batches = 50
out_size = 10

## Dataset definition
experiment_name = 'mnist_test'
videos_to_generate = {'Train':1500, 'Val':500, 'Test':1000}


vaccs, faccs, names = [], [], []

nepochs = 50
base_net = TypicalCNN
metric_list = []
for aggregation in ["poissonbernoulli_aggregation", "tanh_aggregation" , "max_aggregation"]: #"sigmoid_aggregation"

    np.random.seed(1234)
    theano_dataset = get_theano_dataset(experiment_name, videos_to_generate, frames_per_video)
    mynet = Network(video_batches, frames_per_video, out_size, aggregation, base_net, theano_dataset)
    metrics = mynet.train(nepochs, theano_dataset, ['Train','Val'])
    metric_list.append(metrics)
    names.append(aggregation)

from pylab import *

colors = ['r','g','b','c','y']

figure()
for color, name, vacc in zip(colors,names,metric_list):
    plot(vacc['Train']['loss'], '--' + color, label = name + " train")
    plot(vacc['Val']['loss'], '-'  + color, label = name + " val")
ylim([-0.05,0.2])
title("Los")
legend(loc='upper right')


figure()
for color, name, vacc in zip(colors,names,metric_list):
    plot(vacc['Train']['video_acc'], '--' + color, label = name + " train")
    plot(vacc['Val']['video_acc'], '-'  + color, label = name + " val")
ylim([0.2,1.])
title("Video accuracy")
legend(loc='lower right')

colors = ['r','g','b','c','y']

figure()
for color, name, vacc in zip(colors,names,metric_list):
    plot(vacc['Train']['frame_acc'], '--' + color, label = name + " train")
    plot(vacc['Val']['frame_acc'], '-'  + color, label = name + " val")
ylim([0.2,1.])
title("Frame  accuracy")
legend(loc='lower right')



print("DONE")
show()