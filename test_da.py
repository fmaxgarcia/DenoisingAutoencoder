from DenoisingAutoencoder import DenoisingAutoencoder
import theano
import numpy as np
from load_data import load_data

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 15
BATCH_SIZE = 20
DATASET = '../Datasets/mnist.pkl.gz'

import theano.tensor as T
import timeit

if __name__ == '__main__':

    datasets = load_data(DATASET)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE
    np_data = train_set_x.get_value()

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    da = DenoisingAutoencoder(n_visible=28 * 28, n_hidden=500, batch_size=BATCH_SIZE)

    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in range(TRAINING_EPOCHS):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            train_minibatch = np_data[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
            c.append( da.train(train_minibatch, corruption_level=0.0) )

        print('Training epoch %d, cost ' % epoch, np.mean(c))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    # start-snippet-3
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    da = DenoisingAutoencoder(n_visible=28 * 28, n_hidden=500, batch_size=BATCH_SIZE)

    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in range(TRAINING_EPOCHS):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            train_minibatch = np_data[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
            c.append( da.train(train_minibatch, corruption_level=CORRUPTION_LEVEL) )

        print('Training epoch %d, cost ' % epoch, np.mean(c))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)
