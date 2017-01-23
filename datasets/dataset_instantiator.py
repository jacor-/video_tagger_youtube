from datasets.base_datasets.MNISTsequence import MNISTOriginalDataset
from datasets.base_datasets.CIFARsequence10 import CIFAR10OriginalDataset
from datasets.base_datasets.CIFARsequence100 import CIFAR100OriginalDataset
from datasets.dataset_creators.SequenceCreator import SequencesCollector, SequencesTheano
from datasets.base_datasets.YoutubeDataset import YoutubeDataset, YoutubeVideoBaseDataset


def  instantiate_dataset(dataset_name, frames_per_video, videos_to_generate, experiment_name):
    ## Sequences artificially created based on a non-sequence datased
    if dataset_name in ['mnist', 'cifar-100', 'cifar-10']:
        # First we need the original dataset
        if dataset_name == 'mnist':
            original_dataset = MNISTOriginalDataset()
            inp_shape, out_shape = [1,28,28], 10
        elif dataset_name == 'cifar-100':
            original_dataset = CIFAR100OriginalDataset()
            inp_shape, out_shape = [3,32,32], 100
        elif dataset_name == 'cifar-10':
            original_dataset = CIFAR10OriginalDataset()
            inp_shape, out_shape = [3,32,32], 10
        else:
            raise Exception("Dataset not recognised...")
        # Then we artificially build sequences
        video_collector = SequencesCollector(original_dataset, experiment_name, videos_to_generate = videos_to_generate, frames_per_video = frames_per_video)
        theano_sequence = SequencesTheano(original_dataset, video_collector.get_dataset())

    elif dataset_name == 'youtube':
        # In this case we have created the dataset before. We use the name of the dataset to properly load it
        min_occurences_accepted, max_occurences_accepted = 25, 5000
        youtube_base = YoutubeVideoBaseDataset(experiment_name, min_occurences_accepted, max_occurences_accepted)
        theano_sequence = YoutubeDataset(youtube_base)
        inp_shape, out_shape = [3,224,224], youtube_base.get_labels_size()
    return theano_sequence, inp_shape, out_shape

if __name__ == '__main__':
    #from datasets.dataset_instantiator import instantiate_dataset
    data, shape, out = instantiate_dataset('youtube', -1, -1, 'youtube_small_experiment')
    print("Output shape : ", data.getData('Train').eval().shape, "  (videos, frames, <image>)")