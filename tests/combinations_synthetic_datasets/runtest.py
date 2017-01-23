













from tests.combinations_synthetic_datasets import args_getter
from network.SequenceNetworkFunctionalities import Network as AggregatorNetwork
from custom_layers.subnetworks.cnn import BaseSimpleCNN_Monoclass, BaseSimpleCNN_Multiclass
from custom_layers.subnetworks.resnets import ResNet_FullPre_Wide_Multiclass, ResNet_FullPre_Wide_Monoclass, ResNet_FullPreActivation_Multiclass, ResNet_FullPreActivation_Monoclass, ResNet_BottleNeck_FullPreActivation_Multiclass, ResNet_BottleNeck_FullPreActivation_Monoclass
from custom_layers.subnetworks.googlenet import Googlenet

from datasets.dataset_instantiator import instantiate_dataset
import numpy as np
import logging
import settings
import datetime

##
# This function returns the class implemeting a specific network identified by a string and a boolean indicating whether we want a multiclass classification or not
# It could be done more modular... but this code is already overengineered enough.
##
def get_base_net(basenetname, multiclass):
    if basenetname == 'simple_cnn':
        BaseNetClass = BaseSimpleCNN_Multiclass if multiclass else BaseSimpleCNN_Monoclass
    elif basenetname == 'resnet_full_wide':
        BaseNetClass = ResNet_FullPre_Wide_Multiclass if multiclass else ResNet_FullPre_Wide_Monoclass
    elif basenetname == 'resnet_full':
        BaseNetClass = ResNet_FullPreActivation_Multiclass if multiclass else ResNet_FullPreActivation_Monoclass
    elif basenetname == 'resnet_bottleneck':
        BaseNetClass = ResNet_BottleNeck_FullPreActivation_Multiclass if multiclass else ResNet_BottleNeck_FullPreActivation_Monoclass
    elif basenetname == 'googlenet':
        BaseNetClass = Googlenet
    else:
        err_message = 'This base net is not implemented yet! (%s)' % basenetname
        logging.error(err_message)
        raise Exception(err_message)
    return BaseNetClass

##
# Given some parameters we will
# - Load the proper dataset with the desired parameters
# - We will run the training process for the argument parameters and store the output in logs to analyze them a posteriori
##
if __name__ == '__main__':

    # Load command line arguments
    inpargs = args_getter.get_arguments()

    # Prepare the logger
    experiment_name = "%s_%s" % (inpargs['experiment_basename'], str(datetime.datetime.now()).replace(" ","_").replace(":","_"))
    logging.basicConfig(filename='%s/logs/%s.log' % (settings.datapath, experiment_name), level=logging.DEBUG)

    # Log parameters
    logging.debug("------ Input parameters -- ")
    for key in inpargs:
        logging.debug("  - parameter: %s --> %s" % (key, str(inpargs[key])))
    logging.debug("-------------------------- ")

    # Instantiate the dataset we are going to use
    dataset_name = inpargs['testing_dataset']
    frames_per_video = inpargs['frames_per_video']
    videos_to_generate = {
                            'Train':inpargs['base_num_train_examples'],
                            'Test' :inpargs['base_num_test_examples']
                         }
    theano_dataset, inp_shape, out_size = instantiate_dataset(dataset_name, frames_per_video, videos_to_generate, experiment_name)

    # Instantiate the network we are going to use and create the top aggregator layer
    video_batches = inpargs['video_batches']
    aggregation_type = inpargs['aggregation']
    base_net = get_base_net(inpargs['base_net_to_use'], inpargs['multiclass'])
    mynet = AggregatorNetwork(video_batches, frames_per_video, out_size, aggregation_type, base_net, theano_dataset, inp_shape)


    metrics = mynet.train(inpargs['nepochs'], theano_dataset, collect_metrics_for = ['Train','Test'], experiment_name = experiment_name, snapshot_epochs =inpargs['snapshot_interval_epochs'], collect_in_multiples_of = inpargs['log_interval_epochs'])
    results_file = "%s/results/%s" % (settings.datapath, experiment_name)

    # Save results
    logging.info("Results being saved on : %s" % results_file)
    np.save(results_file, metrics)
