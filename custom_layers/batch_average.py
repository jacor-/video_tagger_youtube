import lasagne
import lasagne.layers
import numpy as np
import theano
import theano.tensor as T
import lasagne.nonlinearities


class VideoSummaryMaxLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(VideoSummaryMaxLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return T.clip(T.max(input, axis=1), 0.01, 0.99)

    def get_output_shape_for(self, input_shape):
        return input_shape

class VideoSummaryPoissonBernoulli(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(VideoSummaryPoissonBernoulli, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return 1 - T.prod(T.clip(1-input, 0.01, 0.99), axis=1, no_zeros_in_input=True)

    def get_output_shape_for(self, input_shape):
        return input_shape

class VideoSummarySumSigmoidLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(VideoSummarySumSigmoidLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return lasagne.nonlinearities.sigmoid(T.sum(input, axis=1))

    def get_output_shape_for(self, input_shape):
        return input_shape

class VideoSummarySumTanhLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(VideoSummarySumTanhLayer, self).__init__(incoming, **kwargs)

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


