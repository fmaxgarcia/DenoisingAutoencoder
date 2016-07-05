import theano
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import lasagne

class DenoisingAutoencoder():

    def __init__(self, n_visible=784, n_hidden=500, learning_rate=0.1, batch_size=1):
        
        self.learning_rate = learning_rate
        rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.input_layer = lasagne.layers.InputLayer(shape=(batch_size, n_visible))
        self.hidden_layer = lasagne.layers.DenseLayer(self.input_layer, num_units=n_hidden, 
                            nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.Uniform(), b=lasagne.init.Uniform())
        self.output_layer = lasagne.layers.DenseLayer(self.hidden_layer, num_units=n_visible,
                            nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.Uniform(), b=lasagne.init.Uniform())


        self.all_parameters = lasagne.layers.helper.get_all_params(self.output_layer)

        inputs = T.matrix('input')
        self.inputs_shared = theano.shared( np.zeros((batch_size, n_visible), dtype=theano.config.floatX))

        self.y = lasagne.layers.get_output(self.hidden_layer, {self.input_layer : inputs} )
        self.z = lasagne.layers.get_output(self.output_layer, {self.input_layer : inputs})
 
        # L = - T.sum(inputs * T.log(self.z) + (1 - inputs) * T.log(1 - self.z), axis=1)
        error = T.mean( (inputs - self.z)**2 ) 

        givens = { inputs: self.inputs_shared }

        updates = lasagne.updates.nesterov_momentum(error, self.all_parameters, self.learning_rate)
        self._train = theano.function([], [error], updates=updates, givens=givens) 
        self._get_hidden_output = theano.function([], [self.y], givens=givens)
        self._get_reconstruction = theano.function([], [self.z], givens=givens)


    def _get_corrupted_input(self, inputs, corruption_level):
        """binomial will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``
        """
        return np.asarray(np.random.binomial(size=inputs.shape, n=1,
                    p=1 - corruption_level) * inputs, dtype=theano.config.floatX)


    def train(self, inputs, corruption_level=0.0):
        corrupted_input = self._get_corrupted_input(inputs, corruption_level)
        self.inputs_shared.set_value( corrupted_input )
        error = self._train()
        return error

    def get_hidden_outputs(self, inputs):
        inputs = np.asarray(inputs, dtype=theano.config.floatX)
        self.inputs_shared.set_value( inputs )
        return self._get_hidden_output()[0]

    def get_reconstruction(self, inputs):
        self.inputs_shared.set_value( inputs )
        return self._get_reconstruction()[0]
