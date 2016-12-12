from datasets.dataset_creators.MNISTsequence import MNISTOriginalDataset
from datasets.dataset_creators.CIFARsequence10 import CIFAR10OriginalDataset
from datasets.dataset_creators.CIFARsequence100 import CIFAR100OriginalDataset



datapath = '../data'
path_pretrained_weights = datapath + '/network_pretrained_weights'

path_images = datapath + '/images'
path_images_temp = datapath + '/images/images_aux'

path_dataset = datapath + '/dataset'



youtube_dataset = path_dataset + "/dataset_youtube.csv"
youtube_dataset_npy = path_dataset + "/dataset_youtube.npy"



mnist_seq_dataset_train = path_dataset + "/dataset_mnist_seq_train.csv"
mnist_seq_dataset_train_npy = path_dataset + "/dataset_mnist_seq_train.npy"
mnist_seq_dataset_test = path_dataset + "/dataset_mnist_seq_test.csv"
mnist_seq_dataset_test_npy = path_dataset + "/dataset_mnist_seq_test.npy"
mnist_seq_dataset_val = path_dataset + "/dataset_mnist_seq_val.csv"
mnist_seq_dataset_val_npy = path_dataset + "/dataset_mnist_seq_val.npy"

path_mnist_images = datapath + '/images'
path_cifar_images = datapath + '/images'


synthetic_detaset_details = {
    'mnist': {
                'out_size': 10,
                'inp_shape':[1,28,28],
                'generator_class': MNISTOriginalDataset
            },
    'cifar-100': {
                'out_size': 100,
                'inp_shape':[3,32,32],
                'generator_class': CIFAR100OriginalDataset
            },
    'cifar-10': {
                'out_size': 10,
                'inp_shape':[3,32,32],
                'generator_class': CIFAR10OriginalDataset
            }
}
