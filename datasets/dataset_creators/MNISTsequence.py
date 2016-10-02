
#!/usr/bin/python
import settings as st
import numpy as np
import urllib
import requests
import os.path

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from subprocess import check_output
import gzip
import pickle
import theano
import numpy
import theano.tensor as T

###
## This script will prepare a sequence dataset based on MNIST.
###
class MNISTOriginalDataset(object):
    @staticmethod
    def download_mnist(dataset):
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

    def getNumSamples(self, set_name):
        if set_name == 'Train':
            return self.train_set_y.shape[0]
        elif set_name == 'Val':
            return self.valid_set_y.shape[0]
        elif set_name == 'Test':
            return self.test_set_y.shape[0]
        return 0

    def __init__(self):
        MNISTOriginalDataset.download_mnist(st.path_mnist_images + "/mnist.pkl.gz")
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





class MNISTSequencesCollector(object):
    def _generate_videos_(self, mnist_dataset, videos_to_generate):
        self.tags = {}
        self.frames = {}
        self.video_ids = {}
        for set_name in videos_to_generate:
            self.tags[set_name] = {}
            self.frames[set_name] = {}
            self.video_ids[set_name] = []

        for set_name in videos_to_generate:
            data, labels = mnist_dataset.getSet(set_name)
            labels = labels.eval()
            samples_available = labels.shape[0] - 1
            for i in range(videos_to_generate[set_name]):
                video_id = "video_" + str(i)
                chosen_indexs = [random.randint(0, samples_available-1) for cur_ind in range(self.frames_per_video)]
                self.tags[set_name][video_id] = list(set([np.argmax(labels[j]) for j in chosen_indexs]))
                self.frames[set_name][video_id] = chosen_indexs
                self.video_ids[set_name].append(video_id)

    def __init__(self, experiment_name, videos_to_generate, frames_per_video = 20, save_final_dataset_version = True):
        '''
        If a dataset with name "experiment_name" exists, we load it.
        Otherwise, we use search criteria to retrieve the youtube videos
        '''
        load_again = False
        is_frames = os.path.isfile(st.path_dataset + "/" + experiment_name + "_frames.npy")
        is_tags = os.path.isfile(st.path_dataset + "/" + experiment_name + "_tags.npy")
        is_ids = os.path.isfile(st.path_dataset + "/" + experiment_name + "_video_ids.npy")
        if is_frames and is_tags and is_ids:
            ## We load
            self.frames = np.load(st.path_dataset + "/" + experiment_name + "_frames.npy").item()
            self.tags = np.load(st.path_dataset + "/" + experiment_name + "_tags.npy").item()
            self.video_ids = np.load(st.path_dataset + "/" + experiment_name + "_video_ids.npy").item()
            for key in videos_to_generate:
                for fr in self.frames[key].values():
                    if len(fr) != frames_per_video:
                        load_again = True

            keys_to_generate = sorted(self.frames.keys()) == sorted(videos_to_generate.keys())

            for key in videos_to_generate:
                if len(self.video_ids[key]) != videos_to_generate[key]:
                    load_again = True


        else:
            load_again = True

        if load_again:
            print("Reloading")
            ## We generate

            mnist_dataset = MNISTOriginalDataset()

            self.videos_to_generate = videos_to_generate
            self.frames_per_video = frames_per_video

            self._generate_videos_(mnist_dataset, videos_to_generate)

            if save_final_dataset_version:
                np.save(st.path_dataset + "/" + experiment_name + "_frames.npy", self.frames)
                np.save(st.path_dataset + "/" + experiment_name + "_tags.npy", self.tags)
                np.save(st.path_dataset + "/" + experiment_name + "_video_ids.npy", np.array(self.video_ids))
            del mnist_dataset

    def get_labels(self, video_id, set_name):
        return self.tags[set_name][video_id]

    def get_frames(self, video_id, set_name):
        return self.frames[set_name][video_id]

    def get_available_videos(self, set_name):
        return self.video_ids[set_name]

    def get_dataset(self):
        res = {}
        res['tags'] = self.tags
        res['frames'] = self.frames
        res['video_ids'] = self.video_ids
        return res



class MNISTSequencesTheano(object):
    @staticmethod
    def sharedIndexs2Frames(dataset, set_name):
        # We will choose some indexs accessing the list index_list. This is the list we will reshuffle when we change from an epoch to the next
        # We will use the indexs chosen in the previous step to access the indexs2frames, which will return a matrix of indexs  <batch_size, frames_per_video>
        indexs2frames = []
        for video_id in dataset['video_ids'][set_name]:
            indexs2frames.append(dataset['frames'][set_name][video_id])
        index_list = theano.shared(np.array(range(len(indexs2frames)), dtype = np.int32))
        return index_list, theano.shared(np.array(indexs2frames, dtype =np.int32))


    def __init__(self, dataset):
        keys = ['Train', 'Test', 'Val']

        self.dataset = dataset
        self.frames_per_video = len(dataset['frames']['Train'].values()[0])

        self.mnist_dataset = MNISTOriginalDataset()
        self.indexs = {}
        for key in keys:
            indexs, index_to_train = MNISTSequencesTheano.sharedIndexs2Frames(dataset, key)
            self.indexs[key] = {}
            self.indexs[key]['indexs'] = indexs
            self.indexs[key]['indexs2frames'] = index_to_train

        self.val_indexs = MNISTSequencesTheano.sharedIndexs2Frames(dataset, 'Val')
        self.test_indexs = MNISTSequencesTheano.sharedIndexs2Frames(dataset, 'Test')

        t_index = theano.tensor.scalar(dtype = 'int32')
        t_bsize = theano.tensor.scalar(dtype = 'int32')
        batch_start = t_index * t_bsize
        batch_end = (t_index+1) * t_bsize

        self._get_batch_data = {}
        self._get_batch_video_labels = {}
        self._get_batch_frame_labels = {}
        for key in keys:
            batch_index_tensor = self.indexs[key]['indexs'][batch_start:batch_end]
            frames_tensor = self.mnist_dataset.getData(key)[self.indexs[key]['indexs2frames'][batch_index_tensor]]
            frame_labels_tensor = self.mnist_dataset.getLabels(key)[self.indexs[key]['indexs2frames'][batch_index_tensor]]
            video_labels_tensor = frame_labels_tensor.max(axis=1)
            #self._get_batch[key] = theano.function([t_index, t_bsize], [frames_tensor, labels_tensor])
            self._get_batch_data[key]         = frames_tensor
            self._get_batch_frame_labels[key] = frame_labels_tensor
            self._get_batch_video_labels[key] = video_labels_tensor

        self.input_tensors = {'index':t_index, 'bsize': t_bsize}

    def get_num_batches(self, set_name, videos_per_batch):
        return int(len(self.dataset['video_ids'][set_name]) / videos_per_batch)

    def get_tensor_batch_data(self, set_name):
        return self._get_batch_data[set_name]

    def get_tensor_batch_video_labels(self, set_name):
        return self._get_batch_video_labels[set_name]

    def get_tensor_batch_frame_labels(self, set_name):
        return self._get_batch_frame_labels[set_name]

    def get_input_tensors(self):
        return self.input_tensors

    def shuffle_data(self, set_name):
        index_in_set = np.array(range(len(self.dataset['video_ids'][set_name])), dtype = np.int32)
        np.random.shuffle(index_in_set)
        self.indexs[set_name]['indexs'].set_value(index_in_set)


def aux_visualize_data(theano_sequence, set_name, frames_per_video, batch_size):
    from pylab import *
    data = theano.function(theano_sequence.get_input_tensors().values(), theano_sequence.get_tensor_batch_data(set_name))(batch_size,0)
    video_labels = theano.function(theano_sequence.get_input_tensors().values(), theano_sequence.get_tensor_batch_video_labels(set_name))(batch_size,0)
    frame_labels = theano.function(theano_sequence.get_input_tensors().values(), theano_sequence.get_tensor_batch_frame_labels(set_name))(batch_size,0)

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

def get_theano_dataset(experiment_name, videos_to_generate, frames_per_video):
    mnist_video_collector = MNISTSequencesCollector(experiment_name, videos_to_generate = videos_to_generate, frames_per_video = frames_per_video)
    theano_sequence = MNISTSequencesTheano(mnist_video_collector.get_dataset())
    return theano_sequence

if __name__ == "__main__":
    videos_to_generate = {'Train':8, 'Val':5, 'Test':5}
    frames_per_video = 6
    experiment_name = 'perro'
    mnist_video_collector = MNISTSequencesCollector(experiment_name, videos_to_generate = videos_to_generate, frames_per_video = frames_per_video)
    dataset = mnist_video_collector.get_dataset()
    theano_sequence = MNISTSequencesTheano(dataset)
    #
    aux_visualize_data(theano_sequence, 'Train', frames_per_video, 3)
    theano_sequence.shuffle_data('Train')
    aux_visualize_data(theano_sequence, 'Train', frames_per_video, 3)

    for key_name in ['Train','Test','Val']:
        f = theano.function(theano_sequence.get_input_tensors().values(), theano_sequence.get_tensor_batch_data(key_name))
        f2 = theano.function(theano_sequence.get_input_tensors().values(), theano_sequence.get_tensor_batch_video_labels(key_name))
        f3 = theano.function(theano_sequence.get_input_tensors().values(), theano_sequence.get_tensor_batch_frame_labels(key_name))
        print( [ f(5,0).shape, f2(5,0).shape, f3(5,0).shape ] )
