
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


class MNISTDataset(object):
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
        if set_name == 'Train':
            return self.train_set_x, self.train_set_y
        elif set_name == 'Val':
            return self.valid_set_x, self.valid_set_y
        elif set_name == 'Test':
            return self.test_set_x, self.test_set_y
        return None, None

    def getShape(self, set_name):
        if set_name == 'Train':
            return self.train_set_y.shape[0]
        elif set_name == 'Val':
            return self.valid_set_y.shape[0]
        elif set_name == 'Test':
            return self.test_set_y.shape[0]
        return 0

    def __init__(self):
        MNISTDataset.download_mnist(st.path_mnist_images + "/mnist.pkl.gz")
        train_set, valid_set, test_set = MNISTDataset.load_data(st.path_mnist_images + "/mnist.pkl.gz")

        self.test_set_x, self.test_set_y = MNISTDataset.shared_dataset(test_set)
        self.valid_set_x, self.valid_set_y = MNISTDataset.shared_dataset(valid_set)
        self.train_set_x, self.train_set_y = MNISTDataset.shared_dataset(train_set)




class MNISTSequencesCollector(object):
    def _generate_videos_(self, mnist_dataset):
        self.tags = {}
        self.frames = {}
        self.video_ids = []

        data, labels = mnist_dataset.getSet(self.set)
        labels = labels.eval()
        samples_available = labels.shape[0] - 1
        for i in range(self.videos_to_generate):
            video_id = "video_" + str(i)
            chosen_indexs = [random.randint(0, samples_available-1) for cur_ind in range(self.frames_per_video)]
            self.tags[video_id] = list(set([labels[j] for j in chosen_indexs]))
            self.frames[video_id] = chosen_indexs
            self.video_ids.append(video_id)

    def __init__(self, experiment_name, set, videos_to_generate = 5000, frames_per_video = 20, save_final_dataset_version = True):
        '''
        If a dataset with name "experiment_name" exists, we load it.
        Otherwise, we use search criteria to retrieve the youtube videos
        '''
        mnist_dataset = MNISTDataset()

        self.set = set
        self.videos_to_generate = videos_to_generate
        self.frames_per_video = frames_per_video

        self._generate_videos_(mnist_dataset)

        if save_final_dataset_version:
            np.save(st.path_dataset + "/" + experiment_name + "_tags.npy", self.tags)
            np.save(st.path_dataset + "/" + experiment_name + "_video_ids.npy", np.array(self.video_ids))

    def get_labels(self, video_id):
        return self.tags[video_id]

    def get_frames(self, video_id):
        return self.frames[video_id]

    def get_available_videos(self):
        return self.video_ids

    def save_in_vilynx_format(self, output_file):
        f = open(output_file, "w")
        for video_id in self.get_available_videos():
            for frame in self.get_frames(video_id):
                video_frame = str(frame)
                labels = self.get_labels(video_id)
                f.write(video_frame + ";;;{"+','.join(map(str,labels))+"};"+video_id+"\n")
        f.close()


if __name__ == "__main__":
    mnist_video_collector = MNISTSequencesCollector("mnist_test_experiment_train", 'Train', videos_to_generate = 10000, frames_per_video = 3)
    mnist_video_collector.save_in_vilynx_format(st.mnist_seq_dataset_train)

    mnist_video_collector = MNISTSequencesCollector("mnist_test_experiment_train", 'Val', videos_to_generate = 10, frames_per_video = 3)
    mnist_video_collector.save_in_vilynx_format(st.mnist_seq_dataset_val)

    mnist_video_collector = MNISTSequencesCollector("mnist_test_experiment_train", 'Test', videos_to_generate = 50, frames_per_video = 3)
    mnist_video_collector.save_in_vilynx_format(st.mnist_seq_dataset_test)
