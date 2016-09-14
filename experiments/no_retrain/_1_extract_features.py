from network.base_networks.googlenet import GooglenetNetwork
import lasagne
import numpy as np
from tools.load_images import loadImage
from datasets.video_multitag_dataset import 


images_base_path = ''

minimum_samples_per_tag = 20
minimum_tags_per_video = 2
frames_per_video = 60
dataset = VideoMultitagDataset('datasets/sample_dataset_big.csv', minimum_samples_per_tag, minimum_tags_per_video, frames_per_video)


print("starting")
# We will feed the frames into the network video per video
batch_size = frames_per_video
net = GooglenetNetwork(batch_size)
net, classes, mean_image = net.get_network()
output_layer = net['prob']
## We allocate some space here for the input data
batch_data = np.zeros([batch_size,3, 224, 224])
for videoid in dataset.getVIdeoIds():
	for i, frame_partial_path in enumerate(dataset.getFrames(videoid)):
		batch_data[i,:,:,:] = loadImage( images_base_path + '/' + frame_partial_path)
	video_tags = dataset.getLabels(videoid)
	classification_result = lasagne.layers.get_output(output_layer, batch_data).eval()
	break

