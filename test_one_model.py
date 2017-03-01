from train_model import train_model

params = {}
params['data_file'] = 'Temperature_100_crop_100_poi_90-360.hdf5'
params['x_size'] = 100
params['y_size'] = 100
params['nb_epoch'] = 1
params['btch_sz'] = 10
params['nb_col_1'] = 3
params['nb_col_2'] = 5
params['nb_col_3'] = 5
params['feat_1'] = 32
params['feat_2'] = 64
params['feat_3'] = 128
params['init_method'] = 'glorot_normal'
params['do'] = 0.2
params['bn'] = 'No Batch Norm'
params['relu'] = 'prelu'
params['dense1'] = 256
params['dense2'] = 128
params['dense3'] = 1
params['save'] = False
params['callbacks'] = 'tensorboard'

train_model(params)
