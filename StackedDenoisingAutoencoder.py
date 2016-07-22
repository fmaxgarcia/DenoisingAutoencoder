from DenoisingAutoencoder import *
import theano
import numpy as np
import theano.tensor as T
import lasagne

class TaskType:
    CLASSIFICATION = 0
    REGRESSION = 1

class StackedDenoisingAutoencoder:

    def __init__(self, n_input, n_hidden_list, batch_size):

        n_layers = len(n_hidden_list)
        self.da_layers = []

        for layer_index in range(n_layers):
            da = DenoisingAutoencoder(n_visible=n_input, n_hidden=n_hidden_list[layer_index], batch_size=batch_size, output_nonlinearity=None)
            self.da_layers.append( da )
            n_input = n_hidden_list[layer_index]


    def pre_train(self, train_set_x, epochs, batch_size, corruption_level, corruption_type):

        num_batches = train_set_x.shape[0] / batch_size
        for i in range(len(self.da_layers)):
            da_layer = self.da_layers[i]
            if i == 0:
                for j in range(epochs):
                    c = []
                    for batch_index in range(num_batches):
                        train_minibatch = train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size]
                        c.append( da_layer.train(train_minibatch, corruption_level=corruption_level, corruption_type=corruption_type) )
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
                            c.append( da_layer.train(inputs, corruption_level=corruption_level, corruption_type=corruption_type) )
                    print "Autoencoder %d - Training epoch %d: %f" %(i,j, np.mean(c))


    def build_network(self, num_inputs, num_outputs, learning_rate, batch_size, output_dim,
                    input_type=theano.config.floatX, label_type=theano.config.floatX, task_type=TaskType.CLASSIFICATION):
        self.learning_rate = learning_rate
        self.input_layer = self.da_layers[0].input_layer

        output_nonlinearity = None if task_type == TaskType.REGRESSION else lasagne.nonlinearities.sigmoid

        self.output_layer = lasagne.layers.DenseLayer(self.da_layers[-1].hidden_layer, num_units=num_outputs,
                            nonlinearity=output_nonlinearity, W=lasagne.init.Uniform(), b=lasagne.init.Uniform())

        self.inputs_shared = theano.shared( np.zeros((batch_size, num_inputs), dtype=input_type), name='inputs shared')
        self.labels_shared = theano.shared( np.zeros((batch_size, output_dim), dtype=label_type), name='labels shared')
        inputs = T.matrix('input', dtype=input_type)
        labels = T.matrix('labels', dtype=label_type)
        self.all_parameters = lasagne.layers.helper.get_all_params(self.output_layer)
        self.out = lasagne.layers.get_output(self.output_layer, {self.input_layer : inputs} )

        error = -T.mean(T.log(self.out)[T.arange(labels.shape[0]), labels]) if task_type == TaskType.CLASSIFICATION else T.mean((self.out - labels)**2)

        givens = { inputs : self.inputs_shared, labels : self.labels_shared }
        updates = lasagne.updates.sgd(error, self.all_parameters, self.learning_rate)

        self._train = theano.function([], [error], updates=updates, givens=givens)

        self._get_output = theano.function([], [self.out], givens={ inputs :self.inputs_shared} )

    def train(self, inputs, outputs):
        inputs = np.asarray(inputs, dtype=theano.config.floatX)
        outputs = np.asarray(outputs, dtype=theano.config.floatX)
        self.inputs_shared.set_value( inputs )
        self.labels_shared.set_value( outputs )
        error = self._train()
        return error


    def get_output(self, inputs):
        inputs = np.asarray(inputs, dtype=theano.config.floatX)
        self.inputs_shared.set_value( inputs )
        return self._get_output()[0]
        
    def get_reconstruction(self, inputs):
        inputs = np.asarray(inputs, dtype=theano.config.floatX)
        for layer in self.da_layers:
            if layer is self.da_layers[-1]:
                reconstruction = layer.get_reconstruction(inputs)
            else:
                inputs = layer.get_hidden_outputs(inputs)

        return reconstruction



