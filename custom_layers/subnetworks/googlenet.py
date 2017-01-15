######
###
### TODO: REMOVE THIS ONE AND PREPARE IT AS A CUSTOM LAYER / SUBNETWORK. THE CODE IS BASICALLY THE SAME
###       YOU WILL ONLY NEED TO SPECIFY THE PARAMETERS (example code: custom_layers/subnoetworks/cnn.py)
###
#####


from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear, tanh
import lasagne.layers
import settings
import os
import pickle
import os.path

class GooglenetNetwork(object):
    def __init__(self, incoming, out_size, use_pretrained = True):
        # Some parameters for this specific network
        self.weights_url = "https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl"
        self.weights_filename = settings.path_pretrained_weights + "/blvc_googlenet.pkl"

        # We build the model
        self.net = self.build_model(incoming, out_size)

        if use_pretrained:
            # If the weights file does not exist, we download it
            if not os.path.isfile(self.weights_filename):
                self._download_pretrained_weights()

            # Load the weights, set these parameters into the network
            with open(self.weights_filename, 'rb') as f:
                params = pickle.load(f, encoding = 'latin1')
            output_layer = self.net['loss3/classifier']
            lasagne.layers.set_all_param_values(output_layer, params['param values'])
            # We return the network itself, class names and the mean_image

        self.classes = params['synset words']
        self.mean_image = [103.939, 116.779, 123.68]

    def get_network(self):
        return self.net, self.classes, self.mean_image

    def build_model(self, incoming, out_size):
        net = {}
        #net['input'] = InputLayer((batch_size, 3, 224, 224))
        net['conv1/7x7_s2'] = ConvLayer(
            incoming, 64, 7, stride=2, pad=3, flip_filters=False)
        net['pool1/3x3_s2'] = PoolLayer(
            net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
        net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
        net['conv2/3x3_reduce'] = ConvLayer(
            net['pool1/norm1'], 64, 1, flip_filters=False)
        net['conv2/3x3'] = ConvLayer(
            net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
        net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
        net['pool2/3x3_s2'] = PoolLayer(
          net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

        net.update(self.build_inception_module('inception_3a',
                                          net['pool2/3x3_s2'],
                                          [32, 64, 96, 128, 16, 32]))
        net.update(self.build_inception_module('inception_3b',
                                          net['inception_3a/output'],
                                          [64, 128, 128, 192, 32, 96]))
        net['pool3/3x3_s2'] = PoolLayer(
          net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)

        net.update(self.build_inception_module('inception_4a',
                                          net['pool3/3x3_s2'],
                                          [64, 192, 96, 208, 16, 48]))
        net.update(self.build_inception_module('inception_4b',
                                          net['inception_4a/output'],
                                          [64, 160, 112, 224, 24, 64]))
        net.update(self.build_inception_module('inception_4c',
                                          net['inception_4b/output'],
                                          [64, 128, 128, 256, 24, 64]))
        net.update(self.build_inception_module('inception_4d',
                                          net['inception_4c/output'],
                                          [64, 112, 144, 288, 32, 64]))
        net.update(self.build_inception_module('inception_4e',
                                          net['inception_4d/output'],
                                          [128, 256, 160, 320, 32, 128]))
        net['pool4/3x3_s2'] = PoolLayer(
          net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)

        net.update(self.build_inception_module('inception_5a',
                                          net['pool4/3x3_s2'],
                                          [128, 256, 160, 320, 32, 128]))
        net.update(self.build_inception_module('inception_5b',
                                          net['inception_5a/output'],
                                          [128, 384, 192, 384, 48, 128]))

        net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
        net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                             num_units=1000,
                                             nonlinearity=linear)
        net['prob'] = NonlinearityLayer(net['loss3/classifier'],
                                        nonlinearity=softmax)
        return net

    def _download_pretrained_weights(self):
        os.system("wget -O network_weights "  + self.weights_url)
        os.system("mv network_weights " + self.weights_filename)

    def build_inception_module(self, name, input_layer, nfilters):
        # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
        net = {}
        net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
        net['pool_proj'] = ConvLayer(
            net['pool'], nfilters[0], 1, flip_filters=False)

        net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)

        net['3x3_reduce'] = ConvLayer(
            input_layer, nfilters[2], 1, flip_filters=False)
        net['3x3'] = ConvLayer(
            net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

        net['5x5_reduce'] = ConvLayer(
            input_layer, nfilters[4], 1, flip_filters=False)
        net['5x5'] = ConvLayer(
            net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

        net['output'] = ConcatLayer([
            net['1x1'],
            net['3x3'],
            net['5x5'],
            net['pool_proj'],
            ])

        return {'{}/{}'.format(name, k): v for k, v in net.items()}

















class Googlenet(lasagne.layers.Layer):
    def __init__(self, incoming, out_size, **kwargs):
        super(Googlenet, self).__init__(incoming, **kwargs)


        self.subnetwork = GooglenetNetwork(incoming, out_size)

        #We drop the last layer. Que le peten!
        self.name_last_layer = 'pool5/7x7_s1'
        self.out_size = out_size
        self.mean_image = [0,0,0]
        self.net, inner_params = self.build_model(incoming)

        for i, param in enumerate(inner_params):
            self.add_param(param, param.shape.eval(), name=param.name + "_" + str(i))


    def get_output_for(self, input, **kwargs):
        return lasagne.layers.get_output(self.net['googlenetOutput'], inputs=input, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.out_size)

    def get_useful_params(self):
        return {'mean_image': self.mean_image}

    def build_model(self, incoming):
        net = self.subnetwork.net
        net['googlenetOutput'] = DenseLayer(self.subnetwork.net[self.name_last_layer], num_units=self.out_size, nonlinearity=lasagne.nonlinearities.tanh)
        params = lasagne.layers.get_all_params(self.subnetwork.net[self.name_last_layer], trainable=True)

        return net, params



if __name__ == '__main__':
    # (1, 3, 224, 224)
    inp = theano.tensor.tensor4()
    net, classes, mean_image = GooglenetNetwork(inp, -1, use_pretrained = True).get_network()
    test = theano.function([inp], self.net['loss3/classifier'])
    print("Predicted class: ", classes[predicted])

