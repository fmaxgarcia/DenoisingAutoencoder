from StackedDenoisingAutoencoder import StackedDenoisingAutoencoder

import theano
import numpy as np
from load_mnist import load_data
import sys

sys.path.append("../GrassmanianDomainAdaptation/")

from GrassmanianSampling import flow

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 100
BATCH_SIZE = 20
DATASET = '../Datasets/mnist.pkl.gz'

grassmannian_sampling = True
    
import theano.tensor as T
import timeit

if __name__ == '__main__':

    datasets = load_data(DATASET)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE

    train_x = train_set_x.get_value()
    train_y = train_set_y.get_value()

    test_x = test_set_x.get_value()
    test_y = test_set_y.get_value()
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################
    if grassmannian_sampling:
        dimensions = 20*20
        grassmanian_features = flow(train_x, test_x, t=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), k=dimensions, dim_reduction="PCA")
        pre_train = grassmanian_features[0]
        for i in range(1,grassmanian_features.shape[0]):
            pre_train = np.vstack( (pre_train, grassmanian_features[i]) )
        pca = PCA(n_components=dimensions)
        train_x = pca.fit_transform(train_x)
        test_x = pca.fit_transform(test_x)
    else:
        dimensions = train_x.shape[1]
        pre_train = np.vstack( (train_x, test_x) )




    sda = StackedDenoisingAutoencoder(n_input=pre_train.shape[1], n_hidden_list=[200, 50], batch_size=BATCH_SIZE)

    start_time = timeit.default_timer()

    sda.pre_train(pre_train, epochs=2, batch_size=BATCH_SIZE, corruption_level=CORRUPTION_LEVEL)
    num_outputs = np.unique(train_y).shape[0]
    sda.build_network(num_inputs=dimensions, num_outputs=num_outputs, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, label_type='int32')

    for i in range(TRAINING_EPOCHS):
        c = []
        for batch_index in range(n_train_batches):
            train_minibatch = train_x[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
            labels_minibatch = train_y[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
            labels_minibatch = labels_minibatch.reshape( (BATCH_SIZE, 1) )
            
            if train_minibatch.shape[0] == BATCH_SIZE:
                c.append( sda.train(train_minibatch, labels_minibatch) )
        print "Training epoch %d: %f" %(i, np.mean(c))

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)

    correct = 0.0
    total = 0.0
    for batch_index in range(n_test_batches):
        test_minibatch = test_x[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
        labels_minibatch = test_y[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
        if test_minibatch.shape[0] == BATCH_SIZE:
            predictions = sda.get_output(test_minibatch)[0]
            for i in range(predictions.shape[0]):
                pred = np.argmax(predictions[i])
                if pred == labels_minibatch[i]:
                    correct += 1
                total += 1

    print "Correct: ", correct
    print "Total: ", total
    print "Accuracy: ", (correct / total)


