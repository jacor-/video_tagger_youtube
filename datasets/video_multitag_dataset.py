import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
import numpy as np

# We will build a dataset with the following structure:
##
## videos = {
##    video_id_1 : {
##                 'frames': [framepath1, framepath2, ..., framepathN], 
##                 'tags' : [tag1, tag2, ..., tagN]
##               },
##    video_id_2 : {
##                 'frames': [framepath1, framepath2, ..., framepathN], 
##                 'tags' : [tag1, tag2, ..., tagN]
##               }
## }
##
##
minimum_samples_per_tag = 20
minimum_tags_per_video = 2
frames_per_video = 60

class VideoMultitagDataset(object):
	def __init__(self, filename, minimum_samples_per_tag, minimum_tags_per_video, frames_per_video):
		self.minimum_samples_per_tag = minimum_samples_per_tag
		self.minimum_tags_per_video = minimum_tags_per_video
		self.frames_per_video = frames_per_video
		self.videos = self._read_from_file(filename)
		self.label_encoder = self._prepare_mapping_tag_integer()
		self._processFromTags2Labels()
		self._filterVideosWithAtLeast_X_labels()

		## We check that everything went well. Whatever conditions we consider
		self.SanityCheckVideoDatabase()

	def SanityCheckVideoDatabase(self):
		## We consider that there is an error if the number of frames for a video is not correct or the number of tags is not enough
		# This function raise an exception if something does not look good
		key_errors = []
		for key in self.videos.keys():
			if len(self.videos[key]['labels']) < self.minimum_tags_per_video  or len(self.videos[key]['frames']) != self.frames_per_video :
			   key_errors.append(key)
		assert len(key_errors) == 0, "Something wrong happened with the following videos: " + str(key_errors)
		return True

	def _processRow(self, videos, hash, tags, vid):
	    # If this is the first time we see the video, we create a new entry in the videos structure
	    if vid not in videos.keys():
	        videos[vid] = {'frames': [], 'tags': []}
	    # We add the frame and the tags. All the time the tags are the same ones, so for simplicity we only keep the last one
	    videos[vid]['frames'].append(hash)
	    videos[vid]['tags'] = tags
	    
	def _processTags_fromstr2tags_(self, videos):
		## This function transforms the tags form one string to a list of tags (strings). The reason is that the input is just a long string with commas inside, not a proper list
		for key in videos.keys():
			videos[key]['tags'] = videos[key]['tags'][1:-1].split(",")

	def _read_from_file(self, filename):
		videos = {}
		f = open(filename)
		for line in f.readlines():
			#df = pd.read_csv(filename, sep = ';', header = None)
			#_processRow(videos, df.ix[ind][0], df.ix[ind][3], df.ix[ind][4])
			l = line[:-1].split(";")
			self._processRow(videos, l[0], l[3], l[4])
		self._processTags_fromstr2tags_(videos)
		return videos

	def _collect_frequency_tags(self):
		tag_frequency = {}
		for key in self.videos.keys():
			for tag in self.videos[key]['tags']:
				try:
					tag_frequency[tag] += 1
				except:
					tag_frequency[tag] = 1
		return tag_frequency

	def _get_accepted_tags_tags(self, tag_frequency, min_frequency_tags):
		for tag in tag_frequency.keys():
			if tag_frequency[tag] < min_frequency_tags:
				tag_frequency.pop(tag)
		return tag_frequency

	def _prepare_mapping_tag_integer(self):
		btags = self._collect_frequency_tags()
		btags = self._get_accepted_tags_tags(btags, self.minimum_samples_per_tag)
		le = LabelEncoder()
		le.fit(btags.keys())
		return le

	def _processFromTags2Labels(self):
		for key in self.videos.keys():
			accepted_labels = [x for x in self.videos[key]['tags'] if x in self.label_encoder.classes_]
			self.videos[key]['labels'] = self.label_encoder.transform(accepted_labels)

	def _filterVideosWithAtLeast_X_labels(self):
		for key in self.videos.keys():
			if len(self.videos[key]['labels']) < self.minimum_tags_per_video:
				self.videos.pop(key)



minimum_samples_per_tag = 20
minimum_tags_per_video = 2
dataset = VideoMultitagDataset('sample_dataset_big.csv', minimum_samples_per_tag, minimum_tags_per_video, frames_per_video)

'''
t1 = time.time()


videos = _read_from_file('sample_dataset_big.csv')
print(time.time()-t1)

btags = _collect_frequency_tags(videos)
btags = _get_accepted_tags_tags(btags, minimum_samples_per_tag)
label_encoder = _prepare_mapping_tag_integer(btags.keys())
print(time.time()-t1)

videos = _processFromTags2Labels(videos, label_encoder)
print(time.time()-t1)
print("Total videos before filtering " + str(len(videos)))
videos = _filterVideosWithAtLeast_X_labels(videos, minimum_tags_per_video)
print(time.time()-t1)
print("Total videos after filtering; at least X tags " + str(len(videos)))
'''

'''
from settings import settings
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

print(" - Files will be saved here: " + settings['path_for_files'])
minimum_samples_per_tag = 30

df = pd.read_csv(settings['dataset_filename'], sep = ';', header = None)
df.columns = ['hash','summary','position','mid','vid']
# Sanity check 1
frames_per_video = df.groupby(['vid']).count()['hash'].unique()
assert len(frames_per_video) == 1, 'The number of frames per video is not constant'

vid_tag = df.groupby('vid').first()[['mid']].reset_index()
vid_tag['mid'] = vid_tag['mid'].map(lambda x: list(set(x[1:-1].replace("\"","").split(","))))
labels = np.hstack(vid_tag['mid'])

df2 = pd.DataFrame(columns = ['a'])
df2['a'] = labels
aux = df2.a.value_counts()
accepted_labels = aux[aux > settings['minimum_samples_per_tag']].index.values
print(" - The dataset had %d unique labels. The accepted ones where %d" % (len(list(set(labels))), len(accepted_labels)))

le = LabelEncoder()
le.fit(accepted_labels)

vid_tag['labels'] = vid_tag['mid'].map(lambda x: le.transform([y for y in x if y in accepted_labels]))
# Sanity check 2
should_be_empty = [x for x in list(set(np.hstack(vid_tag['labels'].values))) if x not in le.transform(accepted_labels)]
assert len(should_be_empty) == 0, 'Fuck shit!'

vid_tag = vid_tag[vid_tag['labels'].map(len) > 0]
frames_per_video_sorted = df[df.vid.isin(vid_tag.vid.values)].groupby('vid')['hash'].apply(lambda x: sorted(list(x), key = lambda x: int(x.split("_")[1])*100 + int(x.split("_")[2]))).reset_index()



df_dataset = pd.merge(vid_tag, frames_per_video_sorted, on = 'vid').set_index('vid')


dataset = {}
for vid in df_dataset.index.values:
	now_data = df_dataset.ix[vid]
	dataset[str(vid)] = {'labels': now_data['labels'], 'images': now_data['hash']}

np.save(settings['path_for_files'] + "/" + settings['dict_dataset'], dataset)
np.save(settings['path_for_files'] + "/" + settings['processed_labels_2_original_label'], le.classes_)



# Split train and test sets
available_vids = np.array(list(dataset.keys()))
indexs = np.arange(len(available_vids))
np.random.shuffle(indexs)
samples_in_train = int(np.floor(settings['train_size']*len(available_vids)))
train_samples = available_vids[indexs[:samples_in_train]]
test_samples = available_vids[indexs[samples_in_train:]]

f = open(settings['path_for_files'] + "/" + settings['output_file_train'], 'w')
for sample in train_samples:
	f.write(str(sample) + "\n")
f.close()
f = open(settings['path_for_files'] + "/" + settings['output_file_test'], 'w')
for sample in test_samples:
	f.write(str(sample) + "\n")
f.close()
'''

