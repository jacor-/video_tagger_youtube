
#!/usr/bin/python
from pylab import *
import settings as st
import numpy as np
import urllib
import requests
import os.path

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import gzip
import pickle
import theano
import numpy
import theano.tensor as T
from numpy import random
from datasets.dataset_creators.SequenceCreator import SequencesCollector, SequencesTheano

###
## This script will prepare a sequence dataset based on MNIST.
###
class MNISTOriginalDataset(object):
    @staticmethod
    def download_dataset(dataset):
        if (not os.path.isfile(dataset)):
            from six.moves import urllib
            origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, dataset)
            print('... loading data')

    @staticmethod
    def load_data(dataset):
        with gzip.open(dataset, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)
        return train_set, valid_set, test_set

    @staticmethod
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return T.cast(shared_x, 'float32'), T.cast(shared_y, 'float32')

    def getSet(self, set_name):
        return self.dataset[set_name]['X'], self.dataset[set_name]['Y']

    def getData(self, set_name):
        return self.dataset[set_name]['X']

    def getLabels(self, set_name):
        return self.dataset[set_name]['Y']

    @staticmethod
    def oneHotEncoding(data_y):
        labels = np.zeros([data_y.shape[0], 10])
        for i in range(data_y.shape[0]):
            labels[i][data_y[i]] = 1
        return labels

    def __init__(self):
        MNISTOriginalDataset.download_dataset(st.path_mnist_images + "/mnist.pkl.gz")
        train_set, valid_set, test_set = MNISTOriginalDataset.load_data(st.path_mnist_images + "/mnist.pkl.gz")

        test_set = (test_set[0].reshape([-1, 1, 28,28]), MNISTOriginalDataset.oneHotEncoding(test_set[1]))
        test_set_x, test_set_y = MNISTOriginalDataset.shared_dataset(test_set)
        valid_set = (valid_set[0].reshape([-1, 1, 28,28]), MNISTOriginalDataset.oneHotEncoding(valid_set[1]))
        valid_set_x, valid_set_y = MNISTOriginalDataset.shared_dataset(valid_set)
        train_set = (train_set[0].reshape([-1, 1, 28,28]), MNISTOriginalDataset.oneHotEncoding(train_set[1]))
        train_set_x, train_set_y = MNISTOriginalDataset.shared_dataset(train_set)

        self.dataset = {
            'Train': {'X': train_set_x, 'Y':train_set_y},
            'Val'  : {'X': valid_set_x, 'Y':valid_set_y},
            'Test' : {'X': test_set_x, 'Y':test_set_y}
        }

def aux_visualize_data(theano_sequence, set_name, frames_per_video, batch_size):
    data = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_data(set_name))(batch_size,0)
    video_labels = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_video_labels(set_name))(batch_size,0)
    frame_labels = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_frame_labels(set_name))(batch_size,0)

    for i_batch in range(batch_size):
        figure()
        print("Video labels")
        print(video_labels[i_batch])
        for i in range(frames_per_video):
            print("-- frame label")
            print(frame_labels[i_batch][i])
            subplot((frames_per_video+1)*100 + 10 + (i+1))
            imshow(data[i_batch][i][0], cmap = cm.Greys, interpolation = 'None')
    show()


if __name__ == "__main__":
    videos_to_generate = {'Train':6, 'Val':5, 'Test':5}
    frames_per_video = 6
    experiment_name = 'perro'
    original_dataset = MNISTOriginalDataset()
    mnist_video_collector = SequencesCollector(original_dataset, experiment_name, videos_to_generate = videos_to_generate, frames_per_video = frames_per_video)
    theano_sequence = MNISTSequencesTheano(original_dataset, mnist_video_collector.get_dataset())

    aux_visualize_data(theano_sequence, 'Train', frames_per_video, 3)
    theano_sequence.shuffle_data('Train')
    aux_visualize_data(theano_sequence, 'Train', frames_per_video, 3)

    for key_name in ['Train','Test','Val']:
        f = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_data(key_name))
        f2 = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_video_labels(key_name))
        f3 = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_frame_labels(key_name))
        print( [ f(5,0).shape, f2(5,0).shape, f3(5,0).shape ] )
