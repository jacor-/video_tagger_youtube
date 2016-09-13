import lasagne
import lasagne.layers
import numpy as np
import theano
import theano.tensor as T

class BatchAverageLayer(lasagne.layers.Layer):
    def __init__(self, incoming, frames_per_video, **kwargs):
        super(BatchAverageLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        # The shape of the matrix will be < batch_size / frames_per_video , batch_size >
        constant_batch_mapper = np.zeros([self.input_shape[0] / frames_per_video, self.input_shape[0]], dtype = np.float32)
        for i in range(self.input_shape[0] / frames_per_video):
        	# for each video we write the same weights here, so the result of multiplying this will be equivalent to an average
        	for j in range(frames_per_video):
        		constant_batch_mapper[i][i*frames_per_video + j] = 1. / frames_per_video
        self.constant_batch_mapper = theano.shared(constant_batch_mapper)

    def get_output_for(self, input, **kwargs):
        return T.dot(self.constant_batch_mapper, input)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] / frames_per_video, input_shape[1])
