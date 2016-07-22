import numpy as np
from StackedDenoisingAutoencoder import *
import sys
sys.path.append("../GrassmanianDomainAdaptation/")

from GrassmanianSampling import flow

grassmannian_sampling = False
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt

CORRUPTION_LEVEL = 0.4
LEARNING_RATE = 0.0001
TRAINING_EPOCHS = 1000
BATCH_SIZE = 32
DATASET = "../Datasets/Mars/tablet/"


if __name__ == '__main__':
    
    input1 = np.load(DATASET+"instrument1.npy")
    input2 = np.load(DATASET+"instrument2.npy")
    outputs = np.load(DATASET+"labels.npy")

    if grassmannian_sampling:
            dimensions = 400
            grassmanian_subspaces = flow(input1, input2, t=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), k=dimensions, dim_reduction="PCA")
            pre_train, train_x, test_x = None, None, None
            for i in range(grassmanian_subspaces.shape[0]):
               A = grassmanian_subspaces[i]
               if pre_train == None:
                    train_x = input1.dot( A.dot(A.T) )
                    test_x = input2.dot( A.dot(A.T) )
                    pre_train = np.vstack( (train_x, test_x) )
                    train_y = outputs
                    test_y = outputs
               else:
                    train = input1.dot( A.dot(A.T) )
                    test = input2.dot( A.dot(A.T) )

                    ###Extend training and testing with projected data
                    train_x = np.vstack( (train_x, train) )
                    train_y = np.vstack( (train_y, outputs) )

                    test_x = np.vstack( (test_x, test) )
                    test_y = np.vstack( (test_y, outputs) )

                    ###Extend pre-train with projected training and testing data
                    pre_train = np.vstack( (pre_train, train) )
                    pre_train = np.vstack( (pre_train, test) )
    else:
        pca = PCA(n_components=600)
        combined_inputs = np.vstack( (input1, input2) )
        pca.fit( combined_inputs )
        combined_inputs = pca.transform( combined_inputs )
        train_x = combined_inputs[:input1.shape[0]]
        train_y = outputs
        test_x = combined_inputs[input1.shape[0]:]
        test_y = outputs

        dimensions = train_x.shape[1]
        pre_train = np.vstack( (train_x, test_x) )

    n_train_batches = train_x.shape[0] / BATCH_SIZE
    n_test_batches = test_x.shape[0] / BATCH_SIZE
    print "Training autoencoder..."
    sda = StackedDenoisingAutoencoder(n_input=pre_train.shape[1], n_hidden_list=[400, 200], batch_size=BATCH_SIZE)
    sda.pre_train(pre_train, epochs=200, batch_size=BATCH_SIZE, corruption_level=CORRUPTION_LEVEL, corruption_type=CorruptionType.GAUSSIAN)
    num_outputs = train_y.shape[1]
    sda.build_network(num_inputs=dimensions, num_outputs=num_outputs, output_dim=num_outputs, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)

    for i in range(TRAINING_EPOCHS):
        c = []
        for batch_index in range(n_train_batches):
            train_minibatch = train_x[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
            labels_minibatch = train_y[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
            
            if train_minibatch.shape[0] == BATCH_SIZE:
                # print sda.test(train_minibatch, labels_minibatch)
                c.append( sda.train(train_minibatch, labels_minibatch) )
        print "Training epoch %d: %f" %(i, np.mean(c))

    predictions = None
    for batch_index in range(n_train_batches):
        test_minibatch = test_x[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
        
        if test_minibatch.shape[0] == BATCH_SIZE:
            if predictions == None:
                predictions = sda.get_output(test_minibatch)
            else:
                predictions = np.vstack( (predictions, sda.get_output(test_minibatch)))

    test_y = test_y[:predictions.shape[0]]
    print "Transfer MSE ", np.mean( (test_y-predictions)**2 )
            
    xs = np.linspace(0, test_y.shape[0]-1, num=test_y.shape[0])
    for i in range(test_y.shape[1]):
        plt.plot(xs, test_y[:,i], "r", xs, predictions[:,i], "b")
        plt.show()