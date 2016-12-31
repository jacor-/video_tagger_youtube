'''

This model has been adapted based on the code taken from : https://github.com/FlorianMuellerklein/Identity-Mapping-ResNet-Lasagne

The modifications done on top of the base code are:
 1 - Encapsulate the code into lasagne layers so these specific networks can be just used as part of any other pipeline (taking the parameters and anything of interest)
 2 - We create a subversion of the class for non-monoclass networks. This one will be used in real datasets to try to find more than one object within the same input. We'll see how it works!

This module allows you to invoke the following networks encapsulated as layers:
 - class ResNet_FullPre_Wide_Monoclass(incoming, out_size, n = 6, k = 4, **kwargs)
 - class ResNet_FullPre_Wide_Multiclass(incoming, out_size, n = 6, k = 4, **kwargs)
 - class ResNet_FullPreActivation_Monoclass(incoming, out_size, n = 18, **kwargs)
 - class ResNet_FullPreActivation_Multiclass(incoming, out_size, n = 18, **kwargs)
 - class ResNet_BottleNeck_FullPreActivation_Monoclass(incoming, out_size, n = 18, **kwargs)
 - class ResNet_BottleNeck_FullPreActivation_Multiclass(incoming, out_size, n = 18, **kwargs)
'''

import sys
sys.setrecursionlimit(10000)
import lasagne
from lasagne.nonlinearities import rectify, softmax, sigmoid
from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, helper, batch_norm, BatchNormLayer
# for ResNet
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer, ElemwiseSumLayer, NonlinearityLayer, PadLayer, GlobalPoolLayer, ExpressionLayer
from lasagne.init import Orthogonal, HeNormal, GlorotNormal
he_norm = HeNormal(gain='relu')



# ========================================================================================================================
#   RESNET FULL PREACTIVATON NETWORK AS LAYER  ===========================================================================
# ========================================================================================================================

class ResNet_FullPreActivation(lasagne.layers.Layer):
    def __init__(self, incoming, out_size, n = 18, **kwargs):
        super(ResNet_FullPreActivation, self).__init__(incoming, **kwargs)
        self.n = n
        self.name_last_layer = 'probsout'
        self.out_size = out_size
        self.mean_image = [0,0,0]
        self.net, inner_params = self.build_model(incoming)
        for i, param in enumerate(inner_params):
            self.add_param(param, param.shape.eval(), name=param.name + "_" + str(i))


    def get_output_for(self, input, **kwargs):
        return lasagne.layers.get_output(self.net[self.name_last_layer], inputs=input, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.out_size)

    def get_useful_params(self):
        return {'mean_image': self.mean_image}

    def _residual_block_(self, l, increase_dim=False, projection=True, first=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=he_norm))

        # contains the last weight portion, step 6
        conv_2 = ConvLayer(conv_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(bn_pre_relu, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])
        else:
            block = ElemwiseSumLayer([conv_2, l])

        return block


    def build_model(self, incoming):
        net = {}
        params = {}

        l_in = incoming

        l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=he_norm))

        # first stack of residual blocks, output is 16 x 32 x 32
        l = self._residual_block_(l, first=True)
        for _ in range(1,self.n):
            l = self._residual_block_(l)

        # second stack of residual blocks, output is 32 x 16 x 16
        l = self._residual_block_(l, increase_dim=True)
        for _ in range(1,self.n):
            l = self._residual_block_(l)

        # third stack of residual blocks, output is 64 x 8 x 8
        l = self._residual_block_(l, increase_dim=True)
        for _ in range(1,self.n):
            l = self._residual_block_(l)

        bn_post_conv = BatchNormLayer(l)
        bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

        # average pooling
        avg_pool = GlobalPoolLayer(bn_post_relu)

        net['featsout'] = avg_pool
        params = lasagne.layers.get_all_params(net['featsout'], trainable=True)
        return net, params

class ResNet_FullPreActivation_Monoclass(ResNet_FullPreActivation):
    def __init__(self, incoming, out_size, n = 18, **kwargs):
        super(ResNet_FullPreActivation_Monoclass, self).__init__(incoming, out_size, n = n, **kwargs)

    def build_model(self, incoming):
        net, params = super(ResNet_FullPreActivation_Monoclass, self).build_model(incoming)
        net['probsout'] = lasagne.layers.DenseLayer(
                                                    net['featsout'],
                                                    W=HeNormal(),
                                                    num_units=self.out_size,
                                                    nonlinearity=lasagne.nonlinearities.softmax)
        params = lasagne.layers.get_all_params(net['probsout'], trainable=True)
        return net, params

class ResNet_FullPreActivation_Multiclass(ResNet_FullPreActivation):
    def __init__(self, incoming, out_size, n = 18, **kwargs):
        super(ResNet_FullPreActivation_Multiclass, self).__init__(incoming, out_size, n = n, **kwargs)

    def build_model(self, incoming):
        net, params = super(ResNet_FullPreActivation_Multiclass, self).build_model(incoming)
        net['probsout'] = lasagne.layers.DenseLayer(
                                                    net['featsout'],
                                                    W=HeNormal(),
                                                    num_units=self.out_size,
                                                    nonlinearity=lasagne.nonlinearities.sigmoid)
        params = lasagne.layers.get_all_params(net['probsout'], trainable=True)
        return net, params

# ========================================================================================================================



# ========================================================================================================================
#   RESNET BOTTLENECK FULL PREACTIVATON NETWORK AS LAYER    ==============================================================
# ========================================================================================================================
class ResNet_BottleNeck_FullPreActivation(lasagne.layers.Layer):
    ## n and k are parameters of this specific model
    def __init__(self, incoming, out_size, n = 18, **kwargs):
        super(ResNet_BottleNeck_FullPreActivation, self).__init__(incoming, **kwargs)
        self.n = n

        self.name_last_layer = 'probsout'
        self.out_size = out_size
        self.mean_image = [0,0,0]
        self.net, inner_params = self.build_model(incoming)
        for i, param in enumerate(inner_params):
            self.add_param(param, param.shape.eval(), name=param.name + "_" + str(i))


    def get_output_for(self, input, **kwargs):
        return lasagne.layers.get_output(self.net[self.name_last_layer], inputs=input, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.out_size)

    def get_useful_params(self):
        return {'mean_image': self.mean_image}

    def _residual_bottleneck_block_(self, l, increase_dim=False, first=False):
        input_num_filters = l.output_shape[1]

        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
            out_num_filters = out_num_filters * 4
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        bottleneck_filters = out_num_filters / 4

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=bottleneck_filters, filter_size=(1,1), stride=(1,1), nonlinearity=rectify, pad='same', W=he_norm))

        conv_2 = batch_norm(ConvLayer(conv_1, num_filters=bottleneck_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=he_norm))

        # contains the last weight portion, step 6
        conv_3 = ConvLayer(conv_2, num_filters=out_num_filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(bn_pre_relu, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_3, projection])

        elif first:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_3, projection])

        else:
            block = ElemwiseSumLayer([conv_3, l])

        return block

    def build_model(self, incoming):
        net = {}
        params = {}

        l_in = incoming


        # first layer, output is 16x16x16
        l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=he_norm))

        # first stack of residual blocks, output is 64x16x16
        l = self._residual_bottleneck_block_(l, first=True)
        for _ in range(1,self.n):
            l = self._residual_bottleneck_block_(l)

        # second stack of residual blocks, output is 128x8x8
        l = self._residual_bottleneck_block_(l, increase_dim=True)
        for _ in range(1,self.n):
            l = self._residual_bottleneck_block_(l)

        # third stack of residual blocks, output is 256x4x4
        l = self._residual_bottleneck_block_(l, increase_dim=True)
        for _ in range(1,self.n):
            l = self._residual_bottleneck_block_(l)

        bn_post_conv = BatchNormLayer(l)
        bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

        # average pooling
        avg_pool = GlobalPoolLayer(bn_post_relu)
        net['featsout'] = avg_pool
        params = lasagne.layers.get_all_params(net['featsout'], trainable=True)
        return net, params

class ResNet_BottleNeck_FullPreActivation_Monoclass(ResNet_BottleNeck_FullPreActivation):
    def __init__(self, incoming, out_size, n = 18, **kwargs):
        super(ResNet_BottleNeck_FullPreActivation_Monoclass, self).__init__(incoming, out_size, n = n, **kwargs)

    def build_model(self, incoming):
        net, params = super(ResNet_BottleNeck_FullPreActivation, self).build_model(incoming)
        net['probsout'] = lasagne.layers.DenseLayer(
                                                    net['featsout'],
                                                    W=HeNormal(),
                                                    num_units=self.out_size,
                                                    nonlinearity=lasagne.nonlinearities.softmax)
        params = lasagne.layers.get_all_params(net['probsout'], trainable=True)
        return net, params

class ResNet_BottleNeck_FullPreActivation_Multiclass(ResNet_BottleNeck_FullPreActivation):
    def __init__(self, incoming, out_size, n = 18, **kwargs):
        super(ResNet_BottleNeck_FullPreActivation_Multiclass, self).__init__(incoming, out_size, n = n, **kwargs)

    def build_model(self, incoming):
        net, params = super(ResNet_BottleNeck_FullPreActivation, self).build_model(incoming)
        net['probsout'] = lasagne.layers.DenseLayer(
                                                    net['featsout'],
                                                    W=HeNormal(),
                                                    num_units=self.out_size,
                                                    nonlinearity=lasagne.nonlinearities.sigmoid)
        params = lasagne.layers.get_all_params(net['probsout'], trainable=True)
        return net, params

# ========================================================================================================================




# ========================================================================================================================
#   RESNET FULL PREACTIVATON WIDE NETWORK AS LAYER  ======================================================================
# ========================================================================================================================
class ResNet_FullPre_Wide(lasagne.layers.Layer):
    ## n and k are parameters of this specific model
    def __init__(self, incoming, out_size, n = 6, k = 4, **kwargs):
        super(ResNet_FullPre_Wide, self).__init__(incoming, **kwargs)
        self.n = n
        self.k = k

        self.name_last_layer = 'probsout'
        self.out_size = out_size
        self.mean_image = [0,0,0]
        self.net, inner_params = self.build_model(incoming)
        for i, param in enumerate(inner_params):
            self.add_param(param, param.shape.eval(), name=param.name + "_" + str(i))


    def get_output_for(self, input, **kwargs):
        return lasagne.layers.get_output(self.net[self.name_last_layer], inputs=input, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.out_size)

    def get_useful_params(self):
        return {'mean_image': self.mean_image}


    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def _residual_block_(self, l, increase_dim=False, projection=True, first=False, filters=16):
        if increase_dim:
            first_stride = (2,2)
        else:
            first_stride = (1,1)

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=he_norm))

        #dropout = DropoutLayer(conv_1, p=0.3)

        # contains the last weight portion, step 6
        conv_2 = ConvLayer(conv_1, num_filters=filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])

        elif first:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])

        else:
            block = ElemwiseSumLayer([conv_2, l])

        return block

    def build_model(self, incoming):
        net = {}
        params = {}

        n_filters = {0:16, 1:16*self.k, 2:32*self.k, 3:64*self.k}


        # Building the network
        l_in = incoming

        # first layer, output is 16 x 64 x 64
        l = batch_norm(ConvLayer(l_in, num_filters=n_filters[0], filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=he_norm))

        # first stack of residual blocks, output is 32 x 64 x 64
        l = self._residual_block_(l, first=True, filters=n_filters[1])
        for _ in range(1,self.n):
            l = self._residual_block_(l, filters=n_filters[1])

        # second stack of residual blocks, output is 64 x 32 x 32
        l = self._residual_block_(l, increase_dim=True, filters=n_filters[2])
        for _ in range(1,(self.n+2)):
            l = self._residual_block_(l, filters=n_filters[2])

        # third stack of residual blocks, output is 128 x 16 x 16
        l = self._residual_block_(l, increase_dim=True, filters=n_filters[3])
        for _ in range(1,(self.n+2)):
            l = self._residual_block_(l, filters=n_filters[3])

        bn_post_conv = BatchNormLayer(l)
        bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

        # average pooling
        avg_pool = GlobalPoolLayer(bn_post_relu)

        # fully connected layer

        net['featsout'] = avg_pool
        params = lasagne.layers.get_all_params(net['featsout'], trainable=True)
        return net, params

class ResNet_FullPre_Wide_Monoclass(ResNet_FullPre_Wide):
    def __init__(self, incoming, out_size, n = 6, k = 4, **kwargs):
        super(ResNet_FullPre_Wide_Monoclass, self).__init__(incoming, out_size, n = n, k = k, **kwargs)

    def build_model(self, incoming):
        net, params = super(ResNet_FullPre_Wide_Monoclass, self).build_model(incoming)
        net['probsout'] = lasagne.layers.DenseLayer(
                                                    net['featsout'],
                                                    W=HeNormal(),
                                                    num_units=self.out_size,
                                                    nonlinearity=lasagne.nonlinearities.softmax)
        params = lasagne.layers.get_all_params(net['probsout'], trainable=True)
        return net, params

class ResNet_FullPre_Wide_Multiclass(ResNet_FullPre_Wide):
    def __init__(self, incoming, out_size, n = 6, k = 4, **kwargs):
        super(ResNet_FullPre_Wide_Multiclass, self).__init__(incoming, out_size, n = n, k = k, **kwargs)

    def build_model(self, incoming):
        net, params = super(ResNet_FullPre_Wide_Multiclass, self).build_model(incoming)
        net['probsout'] = lasagne.layers.DenseLayer(
                                                    net['featsout'],
                                                    W=HeNormal(),
                                                    num_units=self.out_size,
                                                    nonlinearity=lasagne.nonlinearities.sigmoid)
        params = lasagne.layers.get_all_params(net['probsout'], trainable=True)
        return net, params
# ========================================================================================================================
