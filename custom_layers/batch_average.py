import lasagne
import lasagne.layers
import numpy as np
import theano
import theano.tensor as T
import lasagne.nonlinearities

class BatchAverageLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(BatchAverageLayer, self).__init__(incoming, **kwargs)
        
    def get_output_for(self, input, **kwargs):
        return lasagne.nonlinearities.tanh(T.sum(input, axis=1))

    def get_output_shape_for(self, input_shape):
        return input_shape


class Video2ImagePool(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(Video2ImagePool, self).__init__(incoming, **kwargs)
        video_batches, frames, channels, weight, height = self.input_shape
        self.new_shape = [video_batches*frames, channels, weight, height] 

    def get_output_for(self, input, **kwargs):
        return T.reshape(input, self.new_shape)

    def get_output_shape_for(self, input_shape):
        return self.new_shape

class ImagePoolToVideo(lasagne.layers.Layer):
    def __init__(self, incoming, video_batches, frames_per_video, **kwargs):
        super(ImagePoolToVideo, self).__init__(incoming, **kwargs)
        in_features, out_features = self.input_shape
        self.new_shape = [video_batches, frames_per_video, out_features]

    def get_output_for(self, input, **kwargs):
        return T.reshape(input, self.new_shape)

    def get_output_shape_for(self, input_shape):
        return self.new_shape


