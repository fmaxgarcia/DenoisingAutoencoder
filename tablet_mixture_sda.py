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

    


if __name__ == '__main__':
    
    input1 = np.load(DATASET+"instrument1.npy")
    input2 = np.load(DATASET+"instrument2.npy")
    outputs = np.load(DATASET+"labels.npy")

    norm_input1, norm_input2, mins, maxs = normalize_data(input1, input2)

    dimensions = 400
    grassmanian_subspaces = flow(norm_input1, norm_input2, t=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), k=dimensions, dim_reduction="PCA")

    mixture_of_subspaces = MixtureOfSubspaces(num_subspaces=len(grassmanian_subspaces), num_dimensions=norm_input1.shape[1])
    mixture_of_subspaces.train_mixture(X=norm_input1, Y=outputs[:,2], subspaces=grassmanian_subspaces)

    predictions = mixture_of_subspaces.make_prediction(norm_input2, grassmanian_subspaces)
    print "Prediction MSE ", np.mean((outputs[:,2] - predictions)**2)
            # pre_train, train_x, test_x = None, None, None
            # for i in range(grassmanian_subspaces.shape[0]):
            #    A = grassmanian_subspaces[i]
            #    if pre_train == None:
            #         train_x = input1.dot( A.dot(A.T) )
            #         test_x = input2.dot( A.dot(A.T) )
            #         pre_train = np.vstack( (train_x, test_x) )
            #         train_y = outputs
            #         test_y = outputs
            #    else:
            #         train = input1.dot( A.dot(A.T) )
            #         test = input2.dot( A.dot(A.T) )

            #         ###Extend training and testing with projected data
            #         train_x = np.vstack( (train_x, train) )
            #         train_y = np.vstack( (train_y, outputs) )

            #         test_x = np.vstack( (test_x, test) )
            #         test_y = np.vstack( (test_y, outputs) )

            #         ###Extend pre-train with projected training and testing data
            #         pre_train = np.vstack( (pre_train, train) )
            #         pre_train = np.vstack( (pre_train, test) )
    

    
            
    xs = np.linspace(0, outputs.shape[0]-1, num=outputs.shape[0])
    # for i in range(test_y.shape[1]):
    plt.plot(xs, outputs[:,2], "r", xs, predictions, "b")
    plt.show()