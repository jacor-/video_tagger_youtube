import argparse
import sys

def get_hardcoded_parameters():
    return {
        'experiment_basename':'simple_experiment',
        'base_net_to_use':'simple_cnn',
        'multiclass':False,
        'testing_dataset':'cifar-10',
        'log_interval_epochs':10,
        'snapshot_interval_epochs':25,
        'nepochs':50000,
        'aggregation':"tanh_aggregation", #, "tanh_aggregation" , "max_aggregation"]: #"sigmoid_aggregation"
        'frames_per_video':3,
        'video_batches':50,
        'base_num_train_examples':1000,
        'base_num_test_examples':100
    }

def read_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name',
                        dest='experiment_basename',
                        required=True,
                        help='Name of the experiment (you will need it to check the logs)'
                        )
    parser.add_argument('--basenet',
                        dest='base_net_to_use',
                        required=True,
                        help='Base net to use.'
                        )
    parser.add_argument('--multiclass',
                        dest='multiclass',
                        default=True,
                        help='Specify wether your prediction is expected to be multiclass or not (i.e last layer of the base net is softwax or not).',
                        type=bool
                        )
    parser.add_argument('--dataset',
                        dest='testing_dataset',
                        required=True,
                        help='Dataset you want to test the network with',
                        )
    parser.add_argument('--log_epochs',
                        dest='log_interval_epochs',
                        default=1,
                        help='Number of iterations you will wait until you log test data again',
                        type=int
                        )
    parser.add_argument('--snapshot_epochs',
                        dest='snapshot_interval_epochs',
                        default=25,
                        help='Number of iterations you will wait until you log test data again',
                        type=int
                        )
    parser.add_argument('--nepochs',
                        dest='nepochs',
                        default=50000,
                        help='Number of epochs you will wait before stopping the training',
                        type=int
                        )
    parser.add_argument('--aggregation',
                        dest='aggregation',
                        required=True,
                        help='Aggregation method you will use'
                        )
    parser.add_argument('--base_train_examples',
                        dest='base_num_train_examples',
                        default=2500,
                        type=int
                        )
    parser.add_argument('--base_test_examples',
                        dest='base_num_test_examples',
                        default=500,
                        type=int
                        )
    parser.add_argument('--frames',
                        dest='frames_per_video',
                        default=3,
                        type=int
                        )
    parser.add_argument('--batch',
                        dest='video_batches',
                        default=50,
                        type=int
                        )

    return vars(parser.parse_args())

def get_arguments():
    hardcoded_test = max([1 if arg == '--hardcodedtest' else 0 for arg in sys.argv])
    if hardcoded_test == 1:
        args_ = get_hardcoded_parameters()
    else:
        args_ = read_arguments()
    return args_

## Usage example:
#THEANO_FLAGS="dnn.enabled=True" PYTHONPATH=. python tests/combinations_synthetic_datasets/runtest.py  --name=simple_experiment --basenet=simple_cnn --multiclass=False --dataset=cifar-10 --log_epochs=10 --nepochs=50000 --aggregation=tanh_aggregation --base_train_examples=1000 --base_test_examples=100 --frames=3 --batch=50 --snapshot_epochs=100

