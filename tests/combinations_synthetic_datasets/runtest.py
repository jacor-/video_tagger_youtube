from tests.combinations_synthetic_datasets import args_getter
from network.SequenceNetworkFunctionalities import Network as AggregatorNetwork
from custom_layers.subnetworks.cnn import BaseSimpleCNN_Monoclass, BaseSimpleCNN_Multiclass
from custom_layers.subnetworks.resnets import ResNet_FullPre_Wide_Multiclass, ResNet_FullPre_Wide_Monoclass, ResNet_FullPreActivation_Multiclass, ResNet_FullPreActivation_Monoclass, ResNet_BottleNeck_FullPreActivation_Multiclass, ResNet_BottleNeck_FullPreActivation_Monoclass

from datasets.dataset_creators.SequenceCreator import get_theano_dataset
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
    else:
        err_message = 'This base net is not implemented yet! (%s)' % basenetname
        logging.error(err_message)
        raise Exception(err_message)
    return BaseNetClass

##
# TODO
##
def get_dataset_parameters(inpargs, settings):
    if inpargs['testing_dataset'] in settings.synthetic_detaset_details.keys():
        dataset_details = settings.synthetic_detaset_details[inpargs['testing_dataset']]
        base_dataset = dataset_details['generator_class']()
        out_size = dataset_details['out_size']
        videos_to_generate = {'Train':inpargs['base_num_train_examples'],
                              'Test' :inpargs['base_num_test_examples']}
        inp_shape = dataset_details['inp_shape']
    else:
        err_message = "Experiment not ready yet! (%s)" % inpargs['testing_dataset']
        logging.error(err_message)
        raise Exception(err_message)
    return dataset_details, base_dataset, out_size, videos_to_generate, inp_shape, base_net

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

    # Load some objects which implements parts of our model
    ## Load the base network which will be applied to each frame
    base_net = get_base_net(inpargs['base_net_to_use'], inpargs['multiclass'])
    ## Load the dataset. Remember that our synthetic datasets is built with permutations of a base original dataset
    dataset_details, base_dataset, out_size, videos_to_generate, inp_shape, base_net = get_dataset_parameters(inpargs, settings)
    theano_dataset = get_theano_dataset(base_dataset, experiment_name, videos_to_generate, inpargs['frames_per_video'])

    # Lets train it!
    mynet = AggregatorNetwork(inpargs['video_batches'], inpargs['frames_per_video'], out_size, inpargs['aggregation'], base_net, theano_dataset, inp_shape)

    metrics = mynet.train(inpargs['nepochs'], theano_dataset, collect_metrics_for = ['Train','Test'], experiment_name = experiment_name, snapshot_epochs =inpargs['snapshot_interval_epochs'], collect_in_multiples_of = inpargs['log_interval_epochs'])

    results_file = "%s/results/%s" % (settings.datapath, experiment_name)
    logging.info("Resuts being saved on : %s" % results_file)
    np.save(results_file, metrics)
