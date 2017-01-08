import pickle
from network.base_networks.googlenet import GooglenetNetwork
import lasagne
import settings as st
import numpy as np
from tools.load_images import loadImage
from datasets.video_multitag_dataset import VideoMultitagDataset
from scripts.YoutubeVideosManager import YoutubeVideoCollector

images_base_path = st.path_images

minimum_samples_per_tag = 20
minimum_tags_per_video = 2
frames_per_video = 20
dataset = VideoMultitagDataset(st.youtube_dataset, minimum_samples_per_tag, minimum_tags_per_video, frames_per_video)


print("starting")
# We will feed the frames into the network video per video
batch_size = frames_per_video
net = GooglenetNetwork(batch_size)
net, classes, mean_image = net.get_network()
output_layer = net['loss3/classifier']
## We allocate some space here for the input data
batch_data = np.zeros([batch_size,3, 224, 224], dtype = np.float32)

video_list = dataset.getVideoIds()
npy_file = open(st.youtube_dataset_npy, 'wb')
pickle.dump(video_list, npy_file)
for sss,videoid in enumerate(video_list):
    print(str(sss) + "   -   " + str(len(video_list)))
    for i, frame_partial_path in enumerate(dataset.getFrames(videoid)):
        batch_data[i,:,:,:] = loadImage( images_base_path + '/' + frame_partial_path, mean_image)
    video_tags = dataset.getLabels(videoid)
    classification_result = lasagne.layers.get_output(output_layer, batch_data).eval()
    pickle.dump(classification_result, npy_file)
npy_file.close()


### EXAMPLE TO LOAD THE DATA FROM THE NPY
npy_file = open(st.youtube_dataset_npy, 'rb')
video_list = pickle.load(npy_file)
output = pickle.load(npy_file)
