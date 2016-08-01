from StackedDenoisingAutoencoder import *
import theano
import numpy as np
from load_amazon import load_data
from sklearn.feature_extraction.text import CountVectorizer

import scipy
from scipy import io
import sys
sys.path.append("../GrassmanianDomainAdaptation/")
sys.path.append("../MixtureOfSubspaces/")
sys.path.append("../Visualization/")
from sklearn.decomposition import PCA

from GrassmanianSampling import flow
from MixtureOfSubspaces import MixtureOfSubspaces
from visualize_isomap import visualize_data
import os

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 10
BATCH_SIZE = 32
DATASET = '../Datasets/office images/'

import theano.tensor as T
import timeit

from random import shuffle

def create_projected_data(proj_x, batch_size, sda):   

    n_batches = proj_x.shape[0] / batch_size
    reconstruction_data = None
    for i in range(n_batches):
        train = proj_x[i*batch_size:(i+1)*batch_size]
        if train.shape[0] == batch_size:
            reconstruction = sda.get_reconstruction(train)
            if reconstruction_data == None:
                reconstruction_data = reconstruction
            else:
                reconstruction_data = np.vstack( (reconstruction_data, reconstruction) )
    return reconstruction_data


if __name__ == '__main__':

    domains = os.listdir(DATASET+"amazon/interest_points/")
    original_train_x, original_test_x = [], []
    original_train_y, original_test_y = [], []
    for i, domain in enumerate(domains):
        print "Loading domain ", domain
        directory = DATASET+"amazon/interest_points/"+domain
        for f in os.listdir(directory):
            matfile = scipy.io.loadmat(directory+"/"+f)
            histogram = matfile['histogram']
            original_train_x.append(histogram[0])
            original_train_y.append(i)

        directory = DATASET+"webcam/interest_points/"+domain
        for f in os.listdir(directory):
            matfile = scipy.io.loadmat(directory+"/"+f)
            histogram = matfile['histogram']
            original_test_x.append(histogram[0])
            original_test_y.append(i)

    original_train_x = np.asarray(original_train_x[:100])
    original_train_y = np.asarray(original_train_y[:100])
    original_test_x = np.asarray(original_test_x[:100])
    original_test_y = np.asarray(original_test_y[:100])

    dimensions = 50
    grassmanian_subspaces = flow(original_train_x, original_test_x, t=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), k=dimensions, dim_reduction="PCA")

    num_outputs = np.unique(original_train_y).shape[0]
    proj_train, proj_test = [], []
    for i, subspace in enumerate(grassmanian_subspaces):
        print "Denoising subspace #%d" %(i)
        proj_s = original_train_x.dot( subspace.dot(subspace.T) )
        proj_t = original_test_x.dot( subspace.dot(subspace.T) )
        # visualize_data( np.vstack((proj_s, proj_t)), np.vstack((original_train_y.reshape( (original_train_y.shape[0], 1)), original_test_y.reshape( (original_test_y.shape[0], 1)))) )

        sda =  StackedDenoisingAutoencoder(n_input=proj_s.shape[1], n_hidden_list=[600, 400], batch_size=BATCH_SIZE)
        pre_train = np.vstack( (proj_s, proj_t) )
        sda.pre_train(train_set_x=pre_train, epochs=2, batch_size=BATCH_SIZE, corruption_level=CORRUPTION_LEVEL, corruption_type=CorruptionType.BINOMIAL)
        
        sda.build_network(num_inputs=pre_train.shape[1], num_outputs=num_outputs, output_dim=1, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, task_type=TaskType.CLASSIFICATION, label_type='int32')

        reconstruction_train = create_projected_data(proj_s, BATCH_SIZE, sda)
        reconstruction_test = create_projected_data(proj_t, BATCH_SIZE, sda)

        proj_train.append( reconstruction_train )
        proj_test.append( reconstruction_test )

    print "Creating mixture of subspaces..."
    mixture_of_subspaces = MixtureOfSubspaces(num_subspaces=len(proj_train), proj_dimension=proj_train[0].shape[1], num_outputs=num_outputs,
                                              original_dimensions=original_train_x.shape[1], task_type=TaskType.CLASSIFICATION)

    num_train_samples = proj_train[0].shape[0]
    mixture_of_subspaces.train_mixture(X=original_train_x[:num_train_samples], Y=original_train_y[:num_train_samples], X_proj=proj_train)

    num_test_samples = proj_test[0].shape[0]
    predictions = mixture_of_subspaces.make_prediction(original_test_x[:num_test_samples], proj_test)
    
    correct = 0.0
    total = 0.0
    for i in range(predictions.shape[0]):
        if np.argmax(predictions[i]) == original_test_y[i]:
            correct += 1
        total += 1

    print "Correct: ", correct
    print "Total: ", total
    print "Accuracy: ", (correct / total)

    with open("Results_Office.txt", "a") as myfile:
        accuracy = correct / total
        myfile.write("%s -> %s\nAccuracy: %f \n" %("amazon", "webcam", accuracy))

