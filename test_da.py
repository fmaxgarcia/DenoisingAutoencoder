from DenoisingAutoencoder import DenoisingAutoencoder
import theano
import numpy as np
from load_data import load_data
from utils import tile_raster_images
from PIL import Image

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 5
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
    print("Total training time: ", training_time)


    print("Saving original and reconstructed images")

    tiled_image = tile_raster_images(X=np_data[:100,:], img_shape=(28, 28), 
        tile_shape=(10, 10), tile_spacing=(1, 1))
    image = Image.fromarray(tiled_image)
    image.save('OriginalImage.png')

    num_iter = int(100/BATCH_SIZE)
    reconstruction = np.zeros( np_data[:100,:].shape )
    for i in range(num_iter):
        data = np_data[BATCH_SIZE*i:BATCH_SIZE*(i+1), :]
        rec = da.get_reconstruction(data)
        reconstruction[BATCH_SIZE*i:BATCH_SIZE*(i+1)] = rec


    tiled_image = tile_raster_images(X=reconstruction, img_shape=(28, 28), 
        tile_shape=(10, 10), tile_spacing=(1, 1))
    image = Image.fromarray(tiled_image)
    image.save('ReconstructedImage.png')
