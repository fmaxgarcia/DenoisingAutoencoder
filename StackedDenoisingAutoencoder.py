from DenoisingAutoencoder import DenoisingAutoencoder
import theano
import numpy as np
import theano.tensor as T
import lasagne

class StackedDenoisingAutoencoder:

    def __init__(self, n_input, n_hidden_list, batch_size):

        n_layers = len(n_hidden_list)
        self.da_layers = []

        for layer_index in range(n_layers):
            da = DenoisingAutoencoder(n_visible=n_input, n_hidden=n_hidden_list[layer_index], batch_size=batch_size)
            self.da_layers.append( da )
            n_input = n_hidden_list[layer_index]


    def pre_train(self, train_set_x, epochs, batch_size, corruption_level=0.0):

        num_batches = train_set_x.shape[0] / batch_size
        for i in range(len(self.da_layers)):
            da_layer = self.da_layers[i]
            if i == 0:
                for j in range(epochs):
                    c = []
                    for batch_index in range(num_batches):
                        train_minibatch = train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size]
                        c.append( da_layer.train(train_minibatch, corruption_level=corruption_level) )
                    print "Autoencoder %d - Training epoch %d: %f" %(i,j, np.mean(c))

            else:
                for j in range(epochs):
                    c = []
                    da_layer.hidden_layer.input_layer = self.da_layers[i-1].hidden_layer
                    for batch_index in range(num_batches):
                        train_minibatch = train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size]
                        inputs = train_minibatch
                        for k in range(i):
                            inputs = self.da_layers[k].get_hidden_outputs(inputs)
                        c.append( da_layer.train(inputs, corruption_level=corruption_level) )
                    print "Autoencoder %d - Training epoch %d: %f" %(i,j, np.mean(c))


    def build_network(self, num_inputs, num_outputs, learning_rate, batch_size,
                    input_type=theano.config.floatX, label_type=theano.config.floatX):
        self.learning_rate = learning_rate
        self.input_layer = self.da_layers[0].input_layer

        self.output_layer = lasagne.layers.DenseLayer(self.da_layers[-1].hidden_layer, num_units=num_outputs,
                            nonlinearity=lasagne.nonlinearities.softmax, W=lasagne.init.Uniform(), b=lasagne.init.Uniform())

        self.inputs_shared = theano.shared( np.zeros((batch_size, num_inputs), dtype=input_type), name='inputs shared')
        self.labels_shared = theano.shared( np.zeros((batch_size, 1), dtype=label_type), name='labels shared')
        inputs = T.matrix('input', dtype=input_type)
        labels = T.matrix('labels', dtype=label_type)
        self.all_parameters = lasagne.layers.helper.get_all_params(self.output_layer)
        self.out = lasagne.layers.get_output(self.output_layer, {self.input_layer : inputs} )

        neg_log_like = -T.mean(T.log(self.out)[T.arange(labels.shape[0]), labels])    
        # error = T.sum( T.neq(T.argmax(self.out[T.arange(batch_size)], axis=1), labels[T.arange(batch_size),0]) )
        # mean_error = T.mean(error)

        givens = { inputs : self.inputs_shared, labels : self.labels_shared }
        updates = lasagne.updates.nesterov_momentum(neg_log_like, self.all_parameters, self.learning_rate)

        self._train = theano.function([], [neg_log_like], updates=updates, givens=givens)
        self._get_output = theano.function([], [self.out], givens={ inputs :self.inputs_shared} )

    def train(self, inputs, outputs):
        inputs = np.asarray(inputs, dtype=theano.config.floatX)
        self.inputs_shared.set_value( inputs )
        self.labels_shared.set_value( outputs )
        error = self._train()
        return error

    def get_output(self, inputs):
        inputs = np.asarray(inputs, dtype=theano.config.floatX)
        self.inputs_shared.set_value( inputs )
        return self._get_output()
        





