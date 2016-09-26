
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
## This script will prepare a sequenCe dataset based on MNIST.
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
        return shared_x, T.cast(shared_y, 'int32')

    def getSet(self, set_name):
        return self.dataset[set_name]['X'], self.dataset[set_name]['Y']

    def getData(self, set_name):
        return self.dataset[set_name]['X']

    def getLabels(self, set_name):
        return self.dataset[set_name]['Y']

    def getShape(self, set_name):
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

        test_set_x, test_set_y = MNISTOriginalDataset.shared_dataset(test_set)
        valid_set_x, valid_set_y = MNISTOriginalDataset.shared_dataset(valid_set)
        train_set_x, train_set_y = MNISTOriginalDataset.shared_dataset(train_set)

        self.dataset = {
            'Train': {'X': train_set_x, 'Y':train_set_y}
            'Val'  : {'X': valid_set_x, 'Y':valid_set_y}
            'Test' : {'X': test_set_x, 'Y':test_set_y}
        }

class MNISTSequencesTheano(object):
    @staticmethod
    def sharedIndexs(dataset, set_name):
        indexs = []
        for video_id in dataset['video_ids'][set_name]:
            indexs.append(dataset['frames'][video_id])
        return theano.shared(np.array(indexs), dtype = 'int32')


    def __init__(self, dataset):
        self.dataset = dataset
        self.frames_per_video = len(dataset['frames']['Train'][0])

        self.mnist_dataset = MNISTOriginalDataset()
        self.train_indexs = sharedIndexs(dataset, 'Train')
        self.val_indexs = sharedIndexs(dataset, 'Val')
        self.test_indexs = sharedIndexs(dataset, 'Test')

        self.tensor_index = theano.tensor.int32()
        self._get_batch = {}
        for key in self.dataset.keys():
            self._get_batch[key] = theano.function([self.tensor_index], [self.mnist_dataset.getData[key][self.tensor_index], self.mnist_dataset.getLabels[key][self.tensor_index]])

    def get_num_batches(self, set_name, videos_per_batch):
        return int(len(self.dataset[set_name]['video_ids']) / videos_per_batch)

    def get_batch(self, batch_index, set_name):
        return self.


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
                self.tags[set_name][video_id] = list(set([labels[j] for j in chosen_indexs]))
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

if __name__ == "__main__":
    videos_to_generate = {'Train':8, 'Val':5, 'Test':5}
    frames_per_video = 6
    mnist_video_collector = MNISTSequencesCollector("mnist_test_experiment_train", videos_to_generate = videos_to_generate, frames_per_video = frames_per_video)

    dataset = mnist_video_collector.get_dataset()
    #theano_sequence = MNISTSequencesTheano(dataset)
