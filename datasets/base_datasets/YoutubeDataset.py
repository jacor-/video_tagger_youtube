import theano
import settings as st
import numpy as np
import os.path
from tools.load_images import loadImage
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
#from scripts.TagsTokenizer import TagsGenerator
import numpy as np
from datasets.dataset_creators.YoutubeDatasetDownloader import TagsGenerator
import theano.tensor as T
class VideotagsVocabularyManager(object):

    def __init__(self, tags, min_occurences_accepted, max_occurences_accepted):
        # Each video will have its own tokenizen tags
        aux_tags = {}
        for video_id in tags:
            aux_tags[video_id] = TagsGenerator.get_tokens(tags[video_id])

        # We build the vocabulary and vocabulary count
        vocab = Counter(np.concatenate(aux_tags.values()))

        accepted_vocab = [key for key in vocab if vocab[key] >= min_occurences_accepted and vocab[key] <= max_occurences_accepted]

        self.word_to_label = {}
        for i, accepted_word in enumerate(accepted_vocab):
            self.word_to_label[accepted_word] = i

        self.label_to_word = {}
        for key in self.word_to_label:
            self.label_to_word[self.word_to_label[key]] = key


    def get_word(self,word):
        return self.label_to_word[word]

    def get_label(self,label):
        return self.word_to_label[label]

    def get_list_of_labels(self, list_of_tags):
        # This one will return existin labels (i.e 0,1,2,3,29...)
        list_of_labels = []
        tokens = TagsGenerator.get_tokens(list_of_tags)
        for token in tokens:
            try:
                list_of_labels.append(self.get_label(token))
            except:
                pass
        return list(set(list_of_labels))

class YoutubeVideoBaseDataset(object):
    def __init__(self, experiment_name, min_ocs_tags = 50, max_ocs_tags = 500):
        if not (os.path.isfile(st.path_dataset + "/" + experiment_name + "_tags.npy") and  os.path.isfile(st.path_dataset + "/" + experiment_name + "_video_ids.npy")):
            raise Exception("You must create a youtube dataset with this name. You can use the tool provided in this repo.")

        self.tags = np.load(st.path_dataset + "/" + experiment_name + "_tags.npy").item()
        self.video_ids = list(np.load(st.path_dataset + "/" + experiment_name + "_video_ids.npy"))

        # We can simply change the min and max occurrences without downloaing the whole dataset again!
        min_occurences_accepted=min_ocs_tags
        max_occurences_accepted=max_ocs_tags
        self.vocab_manager = VideotagsVocabularyManager(self.tags, min_occurences_accepted, max_occurences_accepted)
        self.valid_videoids = []
        for video_id in self.video_ids:
            labels = self.get_labels(video_id)
            if len(labels) > 0:
                self.valid_videoids.append(video_id)

        self.frames = {}
        for video_id in self.video_ids:
            self.frames[video_id] = [file for file in os.listdir(st.path_images) if video_id in file]


    def get_labels_size(self):
        return len(self.vocab_manager.label_to_word)

    def get_labels(self, video_id):
        tags_for_a_video = self.tags[video_id]
        labels_for_a_video = self.vocab_manager.get_list_of_labels(tags_for_a_video)
        return labels_for_a_video

    def get_frames(self, video_id):
        return self.frames[video_id]

    def get_available_videos(self):
        return self.valid_videoids



class YoutubeDataset(object):
    def getSet(self, set_name):
        return self.dataset[set_name]['X'], self.dataset[set_name]['Y']

    def getData(self, set_name):
        return self.dataset[set_name]['X']

    def getLabels(self, set_name):
        return self.dataset[set_name]['Y']

    def load_data(self):
        data = []
        for video_id in self.youtube_videos.get_available_videos():
            frames = [loadImage(st.path_images + "/"+ frame)[0] for frame in self.youtube_videos.get_frames(video_id)]
            data.append(frames)
        return np.array(data)

    def oneHotEncoding(self):
        num_vids = len(self.youtube_videos.get_available_videos())
        out_labels = self.youtube_videos.get_labels_size()
        labels = np.zeros([num_vids, out_labels])
        for i,video_id in enumerate(self.youtube_videos.get_available_videos()):
            for label in self.youtube_videos.get_labels(video_id):
                labels[i][label] = 1
        return labels


    @staticmethod
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return T.cast(shared_x, 'float32'), T.cast(shared_y, 'float32')

    def __init__(self, youtube_videos, keys = ['Train']):
        self.youtube_videos = youtube_videos

        data = self.load_data()
        labels = self.oneHotEncoding()
        test_set_x, test_set_y = YoutubeDataset.shared_dataset((data, labels))

        self.dataset = {
            'Train' : {'X': test_set_x, 'Y':test_set_y}
        }
        # Meta will be used to map back label to entity in case it is required
        self.meta = youtube_videos.vocab_manager.label_to_word

        t_index = theano.tensor.scalar(dtype = 'int32')
        t_bsize = theano.tensor.scalar(dtype = 'int32')
        batch_start = t_index * t_bsize
        batch_end = (t_index+1) * t_bsize
        self.input_tensors = {'index':t_index, 'bsize': t_bsize}

        self.indexs = {}
        self._get_batch_data = {}
        self._get_batch_video_labels = {}
        self._get_batch_frame_labels = {}

        key = 'Train'
        self.indexs[key] = theano.shared(np.array(range(len(self.youtube_videos.video_ids)), dtype = np.int32))
        batch_index_tensor = self.indexs[key][batch_start:batch_end]
        self._get_batch_data[key]         = self.dataset[key]['X'][batch_index_tensor]
        self._get_batch_video_labels[key] = self.dataset[key]['Y'][batch_index_tensor]


    def get_num_batches(self, set_name, videos_per_batch):
        return int(len(self.youtube_videos.get_available_videos()) / videos_per_batch)

    def get_tensor_batch_data(self, set_name):
        return self._get_batch_data[set_name]

    def get_tensor_batch_video_labels(self, set_name):
        return self._get_batch_video_labels[set_name]

    def get_tensor_batch_frame_labels(self, set_name):
        raise Exception("In a non-synthetic dataset it does not make any sense to ask for frame labels. You do not know them!")

    def get_input_tensors(self):
        return self.input_tensors

    def shuffle_data(self, set_name):
        #index_in_set = np.array(range(len(self.youtube_videos.video_ids)), dtype = np.int32)
        #np.random.shuffle(index_in_set)
        #self.indexs[set_name].set_value(index_in_set)
        pass
