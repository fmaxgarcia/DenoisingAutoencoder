import numpy as np
from StackedDenoisingAutoencoder import *
import sys
sys.path.append("../GrassmanianDomainAdaptation/")
sys.path.append("../MixtureOfSubspaces/")

from GrassmanianSampling import flow
from MixtureOfSubspaces import MixtureOfSubspaces

grassmannian_sampling = True
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt

CORRUPTION_LEVEL = 0.4
LEARNING_RATE = 0.0001
TRAINING_EPOCHS = 1000
BATCH_SIZE = 32
DATASET = "../Datasets/Mars/tablet/"

def normalize_data(source, target):
    combined_data = np.vstack( (source, target) )
    mins = np.min(combined_data, axis=0)
    maxs = np.max(combined_data, axis=0)
    norm_data = (combined_data - mins) / (maxs - mins)
    return norm_data[:source.shape[0]], norm_data[source.shape[0]:], mins, maxs


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
    
    input1 = np.load(DATASET+"instrument1.npy")
    input2 = np.load(DATASET+"instrument2.npy")
    outputs = np.load(DATASET+"labels.npy")

    norm_input1, norm_input2, mins, maxs = normalize_data(input1, input2)

    dimensions = 400
    grassmanian_subspaces = flow(norm_input1, norm_input2, t=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), k=dimensions, dim_reduction="PCA")

    proj_train, proj_test = [], []
    for i, subspace in enumerate(grassmanian_subspaces):
        print "Denoising subspace #%d" %(i)
        proj_s = input1.dot( subspace.dot(subspace.T) )
        proj_t = input2.dot( subspace.dot(subspace.T) )
        sda =  StackedDenoisingAutoencoder(n_input=proj_s.shape[1], n_hidden_list=[400, 200], batch_size=BATCH_SIZE)
        pre_train = np.vstack( (proj_s, proj_t) )
        sda.pre_train(train_set_x=pre_train, epochs=10, batch_size=BATCH_SIZE, corruption_level=CORRUPTION_LEVEL, corruption_type=CorruptionType.GAUSSIAN)
        
        sda.build_network(num_inputs=pre_train.shape[1], num_outputs=1, output_dim=1, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, task_type=TaskType.REGRESSION)

        reconstruction_train = create_projected_data(proj_s, BATCH_SIZE, sda)
        reconstruction_test = create_projected_data(proj_t, BATCH_SIZE, sda)

        proj_train.append( reconstruction_train )
        proj_test.append( reconstruction_test )
        
    print "Creating mixture of subspaces..."
    mixture_of_subspaces = MixtureOfSubspaces(num_subspaces=len(proj_train), proj_dimension=proj_train[0].shape[1], original_dimensions=norm_input1.shape[1])

    num_train_samples = proj_train[0].shape[0]
    mixture_of_subspaces.train_mixture(X=norm_input1[:num_train_samples], Y=outputs[:num_train_samples,2], X_proj=proj_train)

    num_test_samples = proj_test[0].shape[0]
    predictions = mixture_of_subspaces.make_prediction(norm_input2[:num_test_samples], proj_test)
    print "Prediction MSE ", np.mean((outputs[:num_test_samples,2] - predictions)**2)
    
            
    xs = np.linspace(0, outputs.shape[0]-1, num=outputs.shape[0])
    plt.plot(xs, outputs[:,2], "r", xs, predictions, "b")
    plt.show()