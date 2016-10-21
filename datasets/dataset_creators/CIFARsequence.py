
#!/usr/bin/python
import settings as st
import numpy as np
import urllib
import requests
import os.path
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import gzip
import pickle
import theano
import numpy
import theano.tensor as T
from numpy import random
from SequenceCreator import SequencesCollector, SequencesTheano
from pylab import *

###
## This script will prepare a sequence dataset based on CIFAR-100 (100 classes).
###
class CIFAROriginalDataset(object):
    @staticmethod
    def download_dataset(dataset):
        if (not os.path.isfile(dataset)):
            from six.moves import urllib
            origin = ('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz')
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, dataset)
            os.system('cd %s; tar -xzf %s' % ('/'.join(dataset.split("/")[:-1]), dataset.split("/")[-1] ))

    @staticmethod
    def load_data(dataset):
        def getImagesAndLabels(data_set):
            images = data_set['data'].reshape([-1,3,32,32]).astype('float32')/255
            labels = np.array(data_set['fine_labels'])
            return images, labels

        meta = numpy.load('%s/meta'%dataset)['fine_label_names']
        train_set = getImagesAndLabels(numpy.load('%s/train'%dataset))
        test_set  = getImagesAndLabels(numpy.load('%s/test'%dataset))
        return train_set, test_set, meta

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

    @staticmethod
    def oneHotEncoding(data_y):
        labels = np.zeros([data_y.shape[0], 100])
        for i in range(data_y.shape[0]):
            labels[i][data_y[i]] = 1
        return labels

    def getSet(self, set_name):
        return self.dataset[set_name]['X'], self.dataset[set_name]['Y']

    def getData(self, set_name):
        return self.dataset[set_name]['X']

    def getLabels(self, set_name):
        return self.dataset[set_name]['Y']

    def __init__(self):
        CIFAROriginalDataset.download_dataset(st.path_cifar_images + "/cifar-100-python.tar.gz")
        train_set, test_set, meta = CIFAROriginalDataset.load_data(st.path_cifar_images + "/cifar-100-python")

        test_set_x, test_set_y = CIFAROriginalDataset.shared_dataset((test_set[0], CIFAROriginalDataset.oneHotEncoding(test_set[1])))
        train_set_x, train_set_y = CIFAROriginalDataset.shared_dataset((train_set[0], CIFAROriginalDataset.oneHotEncoding(train_set[1])))

        self.dataset = {
            'Train': {'X': train_set_x, 'Y':train_set_y},
            'Test' : {'X': test_set_x, 'Y':test_set_y}
        }
        # Meta will be used to map back label to entity in case it is required
        self.meta = meta

def aux_visualize_data(theano_sequence, set_name, frames_per_video, batch_size, original_dataset):
    data = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_data(set_name))(batch_size,0)
    video_labels = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_video_labels(set_name))(batch_size,0)
    frame_labels = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_frame_labels(set_name))(batch_size,0)

    for i_batch in range(batch_size):
        print("Batch %d" % i_batch)
        ### We show the tags related to the whole video
        print(" - The tags corresponding to the whole video are (no order): " + str([original_dataset.meta[i] for i,x in enumerate(video_labels[i_batch]) if x == 1]) )
        ### We write, in order, the tags associated to each frame
        tags_per_frame = [original_dataset.meta[np.argmax(frame_labels[i_batch][i])] for i in range(frames_per_video)]
        print(" - The tags corresponding to these images are (in order): " + str(tags_per_frame))

        ### We plot each frame in a video within the same subplot
        figure()
        print(data[i_batch][i].shape)
        for i in range(frames_per_video):
            subplot((frames_per_video+1)*100 + 10 + (i+1))
            imshow(data[i_batch][i].transpose(1,2,0), cmap = cm.Greys, interpolation = 'None')
    show()


if __name__ == "__main__":
    videos_to_generate = {'Train':7, 'Test':5}
    frames_per_video = 4
    experiment_name = 'perro'
    original_dataset = CIFAROriginalDataset()
    mnist_video_collector = SequencesCollector(original_dataset, experiment_name, videos_to_generate = videos_to_generate, frames_per_video = frames_per_video)
    dataset = mnist_video_collector.get_dataset()
    theano_sequence = SequencesTheano(original_dataset, dataset)

    print("As a sanity check: the size of each data collectable from the dataset:")
    for key_name in ['Train','Test']:
        f = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_data(key_name))
        f2 = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_video_labels(key_name))
        f3 = theano.function(list(theano_sequence.get_input_tensors().values()), theano_sequence.get_tensor_batch_frame_labels(key_name))
        print(" - Split %s" % key_name)
        print(" -- Data shape %s" % str(f(5,0).shape))
        print(" -- Video shape %s" % str(f2(5,0).shape))
        print(" -- Frame shape %s" % str(f3(5,0).shape))

    print("We test that the shuffling works as expected. If so, the first an the second time you should see different images.")
    aux_visualize_data(theano_sequence, 'Train', frames_per_video, 3, original_dataset)
    theano_sequence.shuffle_data('Train')
    aux_visualize_data(theano_sequence, 'Train', frames_per_video, 3, original_dataset)
