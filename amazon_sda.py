from StackedDenoisingAutoencoder import StackedDenoisingAutoencoder
import theano
import numpy as np
from load_amazon import load_data
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.append("../GrassmanianDomainAdaptation/")
from sklearn.decomposition import PCA

from GrassmanianSampling import flow

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 100
BATCH_SIZE = 20
DATASET = '../Datasets/amazon reviews/'

import theano.tensor as T
import timeit

from random import shuffle
grassmannian_sampling = True

if __name__ == '__main__':

    vectorizer = CountVectorizer(min_df=2)
    # domains = ["books", "kitchen_&_housewares", "electronics", "dvd"]
    domains = [str(sys.argv[1]), str(sys.argv[2])] #"books", "dvd", "kitchen_&_housewares", "electronics", "dvd"] 
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

    original_train_x = np.asarray(X[:num_samples,:])
    original_train_y = np.asarray(all_ratings)[:num_samples]

    original_test_x = np.asarray(X[num_samples:,:])
    original_test_y = np.asarray(all_ratings)[num_samples:]


    if grassmannian_sampling:
        dimensions = 1000
        grassmanian_subspaces = flow(original_train_x, original_test_x, t=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), k=dimensions, dim_reduction="PCA")
        pre_train, train_x, test_x = None, None, None
        for i in range(grassmanian_subspaces.shape[0]):
           A = grassmanian_subspaces[i]
           if pre_train == None:
                train_x = original_train_x.dot( A.dot(A.T) )
                test_x = original_test_x.dot( A.dot(A.T) )
                pre_train = np.vstack( (train_x, test_x) )
                train_y = original_train_y
                test_y = original_test_y
           else:
                train = original_train_x.dot( A.dot(A.T) )
                test = original_test_x.dot( A.dot(A.T) )

                ###Extend training and testing with projected data
                train_x = np.vstack( (train_x, train) )
                train_y = np.hstack( (train_y, original_train_y) )

                test_x = np.vstack( (test_x, test) )
                test_y = np.hstack( (test_y, original_test_y) )

                ###Extend pre-train with projected training and testing data
                pre_train = np.vstack( (pre_train, train) )
                pre_train = np.vstack( (pre_train, test) )
    else:
        train_x = original_train_x
        train_y = original_train_y
        test_x = original_test_x
        test_y = original_test_y

        dimensions = train_x.shape[1]
        pre_train = np.vstack( (train_x, test_x) )

    # compute number of minibatches for training, validation and testing
    n_train_batches = pre_train.shape[0] / BATCH_SIZE
    n_test_batches = test_x.shape[0] / BATCH_SIZE

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    sda = StackedDenoisingAutoencoder(n_input=pre_train.shape[1], n_hidden_list=[700, 500], batch_size=BATCH_SIZE)

    start_time = timeit.default_timer()

    sda.pre_train(pre_train, epochs=50, batch_size=BATCH_SIZE, corruption_level=CORRUPTION_LEVEL)
    num_outputs = np.unique(train_y).shape[0]
    sda.build_network(num_inputs=pre_train.shape[1], num_outputs=num_outputs, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, label_type='int32')

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
        stdout.write("\rPredicting Batch %d/%d " %(batch_index, n_test_batches))
        stdout.flush()
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

    with open("Results_G.txt", "a") as myfile:
    accuracy = correct / total
        myfile.write("%s -> %s\nAccuracy: %f \n" %(str(sys.argv[1]), str(sys.argv[2]), accuracy))

