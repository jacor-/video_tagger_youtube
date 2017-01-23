
#!/usr/bin/python
from pylab import *

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

class SequencesCollector(object):
    def _generate_videos_(self, original_dataset, videos_to_generate):
        self.tags = {}
        self.frames = {}
        self.video_ids = {}
        for set_name in videos_to_generate:
            self.tags[set_name] = {}
            self.frames[set_name] = {}
            self.video_ids[set_name] = []

        for set_name in videos_to_generate:
            data, labels = original_dataset.getSet(set_name)
            labels = labels.eval()
            samples_available = labels.shape[0] - 1
            for i in range(videos_to_generate[set_name]):
                video_id = "video_" + str(i)
                chosen_indexs = [random.randint(0, samples_available-1) for cur_ind in range(self.frames_per_video)]
                self.tags[set_name][video_id] = list(set([np.argmax(labels[j]) for j in chosen_indexs]))
                self.frames[set_name][video_id] = chosen_indexs
                self.video_ids[set_name].append(video_id)

    def __init__(self, original_dataset, experiment_name, videos_to_generate, frames_per_video = 20, save_final_dataset_version = True):
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
                for fr in list(self.frames[key].values()):
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


            self.videos_to_generate = videos_to_generate
            self.frames_per_video = frames_per_video

            self._generate_videos_(original_dataset, videos_to_generate)

            if save_final_dataset_version:
                np.save(st.path_dataset + "/" + experiment_name + "_frames.npy", self.frames)
                np.save(st.path_dataset + "/" + experiment_name + "_tags.npy", self.tags)
                np.save(st.path_dataset + "/" + experiment_name + "_video_ids.npy", np.array(self.video_ids))


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



class SequencesTheano(object):
    @staticmethod
    def sharedIndexs2Frames(original_dataset, dataset, set_name):
        # We will choose some indexs accessing the list index_list. This is the list we will reshuffle when we change from an epoch to the next
        # We will use the indexs chosen in the previous step to access the indexs2frames, which will return a matrix of indexs  <batch_size, frames_per_video>
        indexs2frames = []
        for video_id in dataset['video_ids'][set_name]:
            indexs2frames.append(dataset['frames'][set_name][video_id])
        index_list = theano.shared(np.array(range(len(indexs2frames)), dtype = np.int32))
        return index_list, theano.shared(np.array(indexs2frames, dtype =np.int32))


    def __init__(self, original_dataset, dataset):
        keys = ['Train', 'Test']

        self.dataset = dataset
        self.frames_per_video = len(list(dataset['frames']['Train'].values())[0])

        self.indexs = {}
        for key in keys:
            indexs, index_to_train = SequencesTheano.sharedIndexs2Frames(original_dataset, dataset, key)
            self.indexs[key] = {}
            self.indexs[key]['indexs'] = indexs
            self.indexs[key]['indexs2frames'] = index_to_train

        t_index = theano.tensor.scalar(dtype = 'int32')
        t_bsize = theano.tensor.scalar(dtype = 'int32')
        batch_start = t_index * t_bsize
        batch_end = (t_index+1) * t_bsize

        self._get_batch_data = {}
        self._get_batch_video_labels = {}
        self._get_batch_frame_labels = {}
        for key in keys:
            batch_index_tensor = self.indexs[key]['indexs'][batch_start:batch_end]
            frames_tensor = original_dataset.getData(key)[self.indexs[key]['indexs2frames'][batch_index_tensor]]
            frame_labels_tensor = original_dataset.getLabels(key)[self.indexs[key]['indexs2frames'][batch_index_tensor]]
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

