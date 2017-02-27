import numpy as np
import datetime
import time
from write_results import write_results_csv
from evaluate_model import evaluate_model_accuracy
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import MaxPooling2D, Convolution2D
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import callbacks

import h5py


def train_model(params):

    print " --- START MODEL TRAINING --- "

    np.random.seed(1337)  # for reproducibility
    data_file = params['data_file']
    x_size = params['x_size']
    y_size = params['y_size']
    init_method = params['init_method']
    nb_epoch = params['nb_epoch']
    btch_sz = params['btch_sz']
    nb_col_1 = params['nb_col_1']
    nb_col_2 = params['nb_col_2']
    nb_col_3 = params['nb_col_3']
    feat_1 = params['feat_1']
    feat_2 = params['feat_2']
    feat_3 = params['feat_3']
    do = params['do']
    bn = params['bn']
    relu = params['relu']
    dense1 = params['dense1']
    dense2 = params['dense2']
    dense3 = params['dense3']
    save = params['save']
    model_callbacks = params['callbacks']

    print 'loading data...'
    dest = '/Users/jeremielequeux/Documents/Git/deep_weather/' \
           'loaded_data/'+data_file
    data = h5py.File(dest)

    X_train = data['X_train'][...]
    Y_train = data['Y_train'][...]
    X_test = data['X_val'][...]
    Y_test = data['Y_val'][...]

    print('X_Train Shape:', X_train.shape)
    print ('Y_Train Shape:', Y_train.shape)
    # /!\ with these shapes new version of Kerias needs
    # "image_dim_ordering": "th" in ~/.keras/keras.json

    print 'normalizing data...'
    mean = np.mean(X_train)
    std = 3*np.std(X_train)  # keep 3*std : normalize data btw -1 and 1
    X_train -= mean
    X_train /= std

    Y_train -= mean
    Y_train /= std

    X_test -= mean
    X_test /= std

    Y_test -= mean
    Y_test /= std

    print "create model..."
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
    model.add(Dense(dense1, init=init_method))

    if relu == 'relu':
        model.add(Activation('relu'))
    else:
        model.add(PReLU())

    if bn == 'Batch Norm':
        model.add(BatchNormalization())

    model.add(Dense(dense2, init=init_method))

    if relu == 'relu':
        model.add(Activation('relu'))
    else:
        model.add(PReLU())

    model.add(Dense(dense3, init=init_method))

    print "compile model..."
    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime(
            '%Y-%m-%d %Hh%Mm%Ss')

    # Add TensorBoard CallBacks for analysis
    if model_callbacks == 'tensorboard':
        myCallBack = callbacks.TensorBoard(log_dir='./logs/'+date,
                                           histogram_freq=1,
                                           write_graph=True,
                                           write_images=False)
        # Train and Evaluate Model
        print "Fit model with tensorboard..."
        hist = model.fit(X_train, Y_train, batch_size=btch_sz,
                         nb_epoch=nb_epoch,
                         shuffle=True,
                         callbacks=[myCallBack],
                         validation_data=(X_test, Y_test))
    else:
        print "Fit model..."
        hist = model.fit(X_train, Y_train, batch_size=btch_sz,
                         nb_epoch=nb_epoch,
                         shuffle=True,
                         validation_data=(X_test, Y_test))

    print "evaluate model..."
    score = model.evaluate(X_test, Y_test)
    print('score: ', score)

    # measure accuracy
    accuracy = evaluate_model_accuracy(model, data, tolerance=1)

    # saving results in CSV
    fieldname = ['date', 'loss', 'score', 'accuracy', 'epoch', 'batch',
                 'train_samples', 'test_samples', 'image_size',
                 'data_loaded', 'init_method',
                 'features_1', 'row_col_1',
                 'features_2', 'row_col_2',
                 'features_3', 'row_col_3',
                 'drop_out', 'dense1', 'dense2',
                 'dense3', 'other'
                 ]
    result_path = './results/results.csv'
    results = {'date': date,
               'loss': hist.history['loss'][nb_epoch-1],
               'score': score,
               'accuracy': accuracy,
               'epoch': nb_epoch,
               'batch': btch_sz,
               'train_samples': X_train.shape[0],
               'test_samples': X_test.shape[0],
               'image_size': str(x_size)+"*"+str(y_size),
               'data_loaded': data_file,
               'init_method': init_method,
               'features_1': feat_1,
               'row_col_1': nb_col_1,
               'features_2': feat_2,
               'row_col_2': nb_col_2,
               'features_3': feat_3,
               'row_col_3': nb_col_3,
               'drop_out': do,
               'dense1': dense1,
               'dense2': dense2,
               'dense3': dense3,
               'other': "Max-Pooling 2*2 - "+relu+" - "+bn
               }

    write_results_csv(result_path, fieldname, results)

    # save model and features
    if save:
        model_file = './models/'+date+'.json'
        weights_file = './weights/'+date+'.h5'

        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)

        model.save_weights(weights_file)
        print("Model saved on "+model_file)
        print("Weights saved on "+weights_file)
    return
