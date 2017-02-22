import numpy as np
import datetime
import time
from write_results import write_results_csv

from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import MaxPooling2D, Convolution2D
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import callbacks
# from keras.optimizers import SGD

import h5py
np.random.seed(1337)  # for reproducibility

print 'loading data...'
data_file = 'weather_crop_100.hdf5'
dest = '/Users/jeremielequeux/Documents/Git/deep_weather/' \
        'loaded_data/'+data_file
data = h5py.File(dest)

# size of source images
x_size = 100
y_size = 100

X_train = data['X_train'][...]
Y_train = data['Y_train'][...]
X_test = data['X_val'][...]
Y_test = data['Y_val'][...]

print ('X_Train Shape:', X_train.shape)
print ('Y_Train Shape:', Y_train.shape)
# /!\ with these shapes new version of Kerias needs
# "image_dim_ordering": "th" in ~/.keras/keras.json

print 'normalizing data...'
mean = np.mean(X_train)
std = 3*np.std(X_train)  # keep 3*std : normalize data btw -1 and 1
print 'mean: %f' % mean
print 'std: %f' % std
X_train -= mean
X_train /= std

Y_train -= mean
Y_train /= std

X_test -= mean
X_test /= std

Y_test -= mean
Y_test /= std

# Model variables
init_method = 'normal'
nb_epoch = 1
btch_sz = 10
nb_col_1 = 3
nb_col_2 = 5
nb_col_3 = 5
feat_1 = 32
feat_2 = 64
feat_3 = 128
do = 0
bn = 'Batch Norm'
relu = 'relu'

# === Loops to test different parameters on the model
# uncomment one of the 2 loops

# === Loop 1 : test Init Meth / Drop-Out / Batch Norm / Prelu-Relu

# testing several initialization methods
for init_method in ['he_normal', 'glorot_normal', 'normal']:
    # testing several Drop-Out values
    for do in [0, 0.2, 0.4]:
        # testing relu and prelu between Dense layers
        for relu in ['relu', 'prelu']:
            # testing with or without Batch Norm btw Dense
            for bn in ['Batch Norm', 'No Batch Norm']:

# === Loop 2 : test model Features and Row/Col
# testing nbr or row and column
# for nb_col_1 in [3, 5]:
#    for nb_col_2 in [3, 5]:
#        for nb_col_3 in [3, 5]:
            # testing different features arrangement
#            for i in range(2):
#                if (i == 0):
#                    feat_1 = 32
#                    feat_2 = 64
#                    feat_3 = 128
#                else:
#                    feat_1 = 64
#                    feat_2 = 128
#                    feat_3 = 256

                model = Sequential()
                # CNN Layer 1
                model.add(Convolution2D(feat_1, nb_col_1, nb_col_1,
                                        input_shape=(1, x_size, y_size),
                                        init=init_method))
                model.add(Activation('relu'))
                model.add(Convolution2D(feat_1,
                          nb_col_1, nb_col_1, init=init_method))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(do))

                # CNN Layer 2
                model.add(Convolution2D(feat_2,
                          nb_col_2, nb_col_2, init=init_method))
                model.add(Activation('relu'))
                model.add(Convolution2D(feat_2,
                          nb_col_2, nb_col_2, init=init_method))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(do))

                # CNN Layer 3
                model.add(Convolution2D(feat_3,
                          nb_col_3, nb_col_3, init=init_method))
                model.add(Activation('relu'))
                model.add(Convolution2D(feat_3,
                          nb_col_3, nb_col_3, init=init_method))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(do))

                # Dense Layers (fully connected)
                model.add(Flatten())
                model.add(Dense(256, init=init_method))

                if relu == 'relu':
                    model.add(Activation('relu'))
                else:
                    model.add(PReLU())

                if bn == 'Batch Norm':
                    model.add(BatchNormalization())

                model.add(Dense(128, init=init_method))

                if relu == 'relu':
                    model.add(Activation('relu'))
                else:
                    model.add(PReLU())

                model.add(Dense(1, init=init_method))

                model.compile(loss='mean_squared_error',
                              optimizer='adam')

                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime(
                        '%Y-%m-%d %Hh%Mm%Ss')

                # Add TensorBoard CallBacks for analysis
                tbCallBack = callbacks.TensorBoard(log_dir='./logs/'+date,
                                                   histogram_freq=1,
                                                   write_graph=True,
                                                   write_images=True)

                # Train and Evaluate Model
                hist = model.fit(X_train, Y_train, batch_size=btch_sz,
                                 nb_epoch=nb_epoch,
                                 shuffle=True, callbacks=[tbCallBack])
                score = model.evaluate(X_test, Y_test)
                print('score: ', score)

                # --- saving results in CSV ---#
                fieldname = ['date', 'loss', 'score', 'epoch', 'batch',
                             'train_samples', 'test_samples', 'data_loaded',
                             'init_method', 'features_1', 'row_col_1',
                             'features_2', 'row_col_2', 'features_3',
                             'row_col_3', 'drop_out', 'dense', 'other'
                             ]
                result_path = './results/results.csv'
                results = {'date': date,
                           'loss': hist.history['loss'][nb_epoch-1],
                           'score': score,
                           'epoch': nb_epoch,
                           'batch': btch_sz,
                           'train_samples': X_train.shape[0],
                           'test_samples': X_test.shape[0],
                           'data_loaded': data_file,
                           'init_method': init_method,
                           'features_1': feat_1,
                           'row_col_1': nb_col_1,
                           'features_2': feat_2,
                           'row_col_2': nb_col_2,
                           'features_3': feat_3,
                           'row_col_3': nb_col_3,
                           'drop_out': do,
                           'dense': "256-128-1",
                           'other': "Max-Pooling 2*2 - "+relu+" - "+bn
                           }

                write_results_csv(result_path, fieldname, results)
