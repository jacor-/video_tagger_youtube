
#!/usr/bin/python
import settings as st
import numpy as np
import urllib
import requests
import os.path

###
## This script will retrieve "videos_per_search" for each given search criteria and the corresponding tags.
## It can be easily extended to also collect title and/or description.
## The ids of the videos and the tags will be saved in different fields
###



class FramesManager(object):

    @staticmethod
    def _get_num_of_frames_per_video(video_id):
        num_frames = len([file for file in os.listdir(st.path_images) if video_id in file])
        return num_frames

    @staticmethod
    def _get_framelist_per_video(video_id):
        return [file for file in os.listdir(st.path_images) if video_id in file]

    @staticmethod
    def _remove_video_frames(video_id):
        for file in os.listdir(st.path_images):
            if video_id in file:
                os.system("rm " + st.path_images + '/' + file)

    @staticmethod
    def download_video_frames(video_id, frames_per_video):
        result = False

        #We make some calculations to be sure we download video enough to take all the samples wee need
        frames_per_second = 0.2
        start_second = 15
        required_frames = frames_per_video
        last_minute = int((required_frames / frames_per_second + start_second) / 60)+1

        #We only download frames if they do not already exist in the proper folder
        if FramesManager._get_num_of_frames_per_video(video_id) != frames_per_video:
            FramesManager._remove_video_frames(video_id)
            os.system("youtube-dl https://www.youtube.com/watch?v={video_id} -o - | avconv -i - -s 240x240 -f image2 -ss 00:00:{first_sec}.00 -t 00:{last_minute}:00.00 -r {samples_per_sec} {path}/{video_id}_%d.jpg".format(path= st.path_images_temp, video_id = video_id, first_sec = "%02d" % start_second, last_minute = "%02d" % last_minute, samples_per_sec = frames_per_second))

            created_files = [x for x in os.listdir(st.path_images_temp) if video_id in x]
            if len(created_files) >= frames_per_video:
                for created_file in created_files[:frames_per_video]:
                    origin_path = st.path_images_temp + "/" + created_file
                    destination_path = st.path_images + "/" + created_file
                    os.system("mv " + origin_path + " " + destination_path)
                result = True

            os.system("rm -rf " + st.path_images_temp + "/*")
        else:
            result = True
        return result, FramesManager._get_framelist_per_video(video_id)



class YoutubeVideoCollector(object):
    def __init__(self, api_key, experiment_name, search_criteria = [], videos_per_search = 50, frames_per_video = 20, save_final_dataset_version = True):
        '''
        If a dataset with name "experiment_name" exists, we load it.
        Otherwise, we use search criteria to retrieve the youtube videos
        '''
        self.api_key = api_key
        self.frames_per_video = frames_per_video
        if os.path.isfile(st.path_dataset + "/" + experiment_name + "_tags.npy") and  os.path.isfile(st.path_dataset + "/" + experiment_name + "_video_ids.npy"):
            print("The dataset already exists. We collect video_ids and tags from files")
            self.tags = np.load(st.path_dataset + "/" + experiment_name + "_tags.npy").item()
            self.video_ids = list(np.load(st.path_dataset + "/" + experiment_name + "_video_ids.npy"))

        else:
            print("The dataset does not exist. We collect video_ids and tags from youtube and save them into files")
            self.tags, self.video_ids = self.collect_video_ids(search_criteria, videos_per_search)
            np.save(st.path_dataset + "/" + experiment_name + "_tags.npy", self.tags)
            np.save(st.path_dataset + "/" + experiment_name + "_video_ids.npy", np.array(self.video_ids))

        self.download_frames_and_filter_incorrects()

        if save_final_dataset_version:
            np.save(st.path_dataset + "/" + experiment_name + "_tags.npy", self.tags)
            np.save(st.path_dataset + "/" + experiment_name + "_video_ids.npy", np.array(self.video_ids))

        min_occurences_accepted=50
        max_occurences_accepted=200
        self.vocab_manager = VideotagsVocabularyManager(self.tags, min_occurences_accepted, max_occurences_accepted)
        self.valid_videoids = []
        for video_id in self.video_ids:
            labels = self.get_labels(video_id)
            if len(labels) > 0:
                self.valid_videoids.append(video_id)


    def get_labels(self, video_id):
        tags_for_a_video = self.tags[video_id]
        labels_for_a_video = self.vocab_manager.get_list_of_labels(tags_for_a_video)
        return labels_for_a_video

    def get_frames(self, video_id):
        return self.frames[video_id]

    def get_available_videos(self):
        return self.valid_videoids

    def save_in_vilynx_format(self, output_file):
        f = open(output_file, "w")
        for video_id in self.get_available_videos():
            for frame in self.get_frames(video_id):
                video_frame = frame
                labels = self.get_labels(video_id)
                f.write(video_frame + ";;;{"+','.join(map(lambda label: self.vocab_manager.get_word(label),labels))+"};"+video_id+"\n")
        f.close()

    def download_frames_and_filter_incorrects(self):
        self.frames = {}
        for video_id in sorted(self.video_ids):
            correct, framelist = FramesManager.download_video_frames(video_id, self.frames_per_video)
            if not correct:
                self.video_ids.remove(video_id)
                self.tags.pop(video_id)
            else:
                self.frames[video_id] = framelist


    def collect_video_ids(self, search_criteria, videos_per_search):
        #print("Expected " + str(len(search_criteria) * videos_per_search) + " videos")
        tags = {}
        for to_search in search_criteria:
            video_, tag_ = YoutubeVideoCollector._retrieve_video_ids_(to_search, videos_per_search, self.api_key)
            for vid, tag in zip(video_, tag_):
                tags[vid] = tag
        video_ids = tags.keys()
        return tags, video_ids

    @staticmethod
    def _get_video_details_(videos_info, videos, tags, api_key):
        for item in videos_info:
            videoid = item['id']['videoId']
            req = requests.get("https://www.googleapis.com/youtube/v3/videos?key={key}&fields=items(snippet(tags))&part=snippet&id={video_id}".format(video_id = videoid, key = api_key))
            try:
               tag = eval(req.content)['items'][0]['snippet']['tags']
            except:
                tag = []
            tags.append(list(tag))
            videos.append(videoid)
        return videos, tags

    @staticmethod
    def _retrieve_video_ids_(search_criteria, videos_per_search, api_key):
        videos, tags = [], []

        p_search_criteria = urllib.quote(search_criteria)
        page_token = ""
        if videos_per_search > 50:
            videos_per_call = 50
        else:
            videos_per_call = videos_per_search

        req = requests.get("https://www.googleapis.com/youtube/v3/search?key={key}&fields=nextPageToken,items(id(videoId))&part=id,snippet&q={query}&maxResults={num_vids}{pageToken}".format(query = p_search_criteria, key = api_key, pageToken = page_token, num_vids = str(videos_per_call)))
        retrieved_videos = eval(req.content)

        # Collect tags for the videos in this page
        videos, tags = YoutubeVideoCollector._get_video_details_(retrieved_videos['items'], videos, tags, api_key)

        # If we do not have enough videos, go to the next page
        while len(videos) < videos_per_search:
            page_token = retrieved_videos['nextPageToken']
            req = requests.get("https://www.googleapis.com/youtube/v3/search?key={key}&fields=nextPageToken,items(id(videoId))&part=id,snippet&q={query}&maxResults={num_vids}{pageToken}".format(query = p_search_criteria, key = api_key, pageToken = "&pageToken="+page_token, num_vids = str(videos_per_call)))
            retrieved_videos = eval(req.content)
            videos, tags = YoutubeVideoCollector._get_video_details_(retrieved_videos['items'], videos, tags, api_key)

        return videos[:videos_per_search], tags[:videos_per_search]



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

# Any results you write to the current directory are saved as output.

import warnings; warnings.filterwarnings("ignore");
import time
start_time = time.time()

from sklearn.ensemble import RandomForestRegressor
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
import re
#import enchant
import random
random.seed(2016)


class TagsGenerator(object):
    ## TODO : delete more stopwords!
    @staticmethod
    def _str_stem(s):
        stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
        strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

        if isinstance(s, str):
            s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
            s = s.lower()
            s = s.replace("  "," ")
            s = s.replace(",","") #could be number / segment later
            s = s.replace("$"," ")
            s = s.replace("?"," ")
            s = s.replace("-"," ")
            s = s.replace("//"," ")
            s = s.replace(")"," ")
            s = s.replace("("," ")
            s = s.replace("..",".")
            s = s.replace(" / "," ")
            s = s.replace(" \\ "," ")
            s = s.replace("."," . ")
            s = s.replace(","," ")
            s = re.sub(r"(^\.|/)", r"", s)
            s = re.sub(r"(\.|/)$", r"", s)
            s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
            s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
            s = s.replace(" x "," xbi ")
            s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
            s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
            s = s.replace("*"," xbi ")
            s = s.replace(" by "," xbi ")
            s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
            s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
            s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
            s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
            s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
            s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
            s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
            s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
            s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
            s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
            s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
            s = s.replace(" v "," volts ")
            s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
            s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
            s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
            s = s.replace("  "," ")
            s = s.replace(" . "," ")
            #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
            s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
            s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
            return s
        else:
            return "null"

    @staticmethod
    def _tokenize(s):
        return s.split(" ")

    @staticmethod
    def get_tokens(original_tag_list):
        unique_tags = []
        for tags in original_tag_list:
            try:
                new_tags = TagsGenerator._tokenize(TagsGenerator._str_stem(tags))
                unique_tags += new_tags
            except:
                pass
        return [x for x in list(set(unique_tags)) if len(x) > 1]


from collections import Counter
#from scripts.TagsTokenizer import TagsGenerator
import numpy as np
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



if __name__ == "__main__":
    videos_per_search = 5000

    api_key = "AIzaSyDV_m-yM9Fdba2rem6w6Cy2GJeWwE2_3r8"
    search_criteria = [#"dog", "cat", "bird", "turtle",
                       #"fight", "run", "walk", "climb",
                       "messi neymar suarez", "ronado benzema bale", "guardiola mourinho simeone","barca chelsea munich"
                       #"guardiola", "mourinho","lebron james", "marc gasol", "pau gasol", "tiger woods", "simeone",
                       #"barcelona", "london", "rome", "amsterdam", "europe", "world",
                       #"plane", "car", "bike",
                       #"mariano rajoy", "pablo iglesias", "pedro sanchez", "albert rivera"
                       ]

    youtube_videos = YoutubeVideoCollector(api_key, "test_experiment", search_criteria = search_criteria, videos_per_search = 100, frames_per_video = 20)
    youtube_videos.save_in_vilynx_format(st.youtube_dataset)







# http://stackoverflow.com/questions/9064962/key-frame-extraction-from-video