from StackedDenoisingAutoencoder import StackedDenoisingAutoencoder
import theano
import numpy as np
from load_amazon import load_data
from sklearn.feature_extraction.text import CountVectorizer

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 100
BATCH_SIZE = 20
DATASET = '../Datasets/amazon reviews/'

import theano.tensor as T
import timeit

from random import shuffle

if __name__ == '__main__':

    vectorizer = CountVectorizer(min_df=2)
    domains = ["books", "kitchen_&_housewares", "electronics", "dvd"] 
    all_ratings, all_text = [], []
    num_samples = 0
    for domain in domains:
        print "loading domain: ", domain
        corpus, ratings = load_data(DATASET, domain)
        indices = range(len(corpus))
        shuffle(indices)
        corpus = np.asarray(corpus)[indices]
        ratings = np.asarray(ratings)[indices]
        all_ratings.extend( list(ratings) )
        all_text.extend( list(corpus) )
        if domain != "dvd":
            num_samples += len(indices) 


    X = vectorizer.fit_transform(all_text)
    X = X.toarray()

    train_x = np.asarray(X[:num_samples,:], dtype=theano.config.floatX)
    train_y = np.asarray(all_ratings, dtype='int32')[:num_samples]

    test_x = np.asarray(X[num_samples:,:], dtype=theano.config.floatX)
    test_y = np.asarray(all_ratings, dtype='int32')[num_samples:]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.shape[0] / BATCH_SIZE
    n_test_batches = test_x.shape[0] / BATCH_SIZE

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    sda = StackedDenoisingAutoencoder(n_input=train_x.shape[1], n_hidden_list=[200, 50], batch_size=BATCH_SIZE)

    start_time = timeit.default_timer()

    sda.pre_train(train_x, epochs=20, batch_size=BATCH_SIZE, corruption_level=CORRUPTION_LEVEL)
    num_outputs = np.unique(train_y).shape[0]
    sda.build_network(num_inputs=train_x.shape[1], num_outputs=num_outputs, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, label_type='int32')

    for i in range(TRAINING_EPOCHS):
        c = []
        for batch_index in range(n_train_batches):
            train_minibatch = train_x[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
            labels_minibatch = train_y[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
            labels_minibatch = labels_minibatch.reshape( (BATCH_SIZE, 1) )
            
            c.append( sda.train(train_minibatch, labels_minibatch) )
        print "Training epoch %d: %f" %(i, np.mean(c))

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)

    correct = 0.0
    total = 0.0
    for batch_index in range(n_test_batches):
        test_minibatch = test_x[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
        labels_minibatch = test_y[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
        predictions = sda.get_output(test_minibatch)[0]
        for i in range(predictions.shape[0]):
            pred = np.argmax(predictions[i])
            if pred == labels_minibatch[i]:
                correct += 1
            total += 1

    print "Correct: ", correct
    print "Total: ", total
    print "Accuracy: ", (correct / total)


