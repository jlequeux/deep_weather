from train_model import train_model

params = {}
params['data_file'] = 'Temperature_200_crop_100_poi_90-360.hdf5'
params['x_size'] = 100
params['y_size'] = 100
params['nb_epoch'] = 10
params['btch_sz'] = 10


for i, j, k, l, m, n in zip((3, 3, 5),
                            (5, 5, 3),
                            (5, 3, 5),
                            (32, 64, 32),
                            (64, 128, 64),
                            (128, 256, 128)):
    params['nb_col_1'] = i
    params['nb_col_2'] = j
    params['nb_col_3'] = k
    params['feat_1'] = l
    params['feat_2'] = m
    params['feat_3'] = n

    for o, p, q, r in zip(('normal', 'glorot_normal', 'he_normal'),
                          (0, 0, 0),
                          ('No Batch Norm', 'No Batch Norm', 'No Batch Norm'),
                          ('relu', 'prelu', 'prelu')):

        params['init_method'] = o
        params['do'] = p
        params['bn'] = q
        params['relu'] = r

        for s, t, u in zip((512, 256, 128),
                           (256, 128, 64),
                           (1, 1, 1)):

            params['dense1'] = s
            params['dense2'] = t
            params['dense3'] = u

            train_model(params)
