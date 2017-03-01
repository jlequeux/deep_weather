import datetime
import h5py
import numpy as np
import os
import pygrib

"""Load noaa data into hd5 X/Y dataset for training."""
"""Crop the images around the point of interest"""

ANL_DATA = '/Volumes/JLE-ExFAT/Deep Learning/weather_data/'
# '/Users/jeremielequeux/Documents/Git/deep_weather/weather_data/'
ANL_LEVEL = 10
TRAIN_SAMPLES = 6000
VAL_SAMPLES = 1000
FEATURE = 'Temperature'
training_ratio = float(TRAIN_SAMPLES) / (VAL_SAMPLES + TRAIN_SAMPLES)
# define point of interest x and y (location where you want to predict)
poi_x = 90
poi_y = 360

# crop the image to get a 100x100 image around  the point of interest
crop_size = 100
# size of original images
x_size = 361
y_size = 720

grib_list = []

d1 = datetime.date(2010, 1, 11)
d2 = datetime.date(2015, 12, 30)

delta = d2 - d1

# destination file
dest = './loaded_data/'+FEATURE+'_'+str(TRAIN_SAMPLES)+'_' \
        'crop_'+str(crop_size)+'_poi_'+str(poi_x)+'-'+str(poi_y)+'.hdf5'

print dest

# crop the X axis and define the position of the poi on the croped image
if poi_x < crop_size//2:
    x_start = 0
    x_end = crop_size
    cpoi_x = poi_x
elif poi_x + crop_size//2 > x_size:
    x_start = x_size - crop_size
    x_end = x_size
    cpoi_x = crop_size - (x_size - poi_x)
else:
    x_start = poi_x - crop_size//2
    x_end = poi_x + crop_size//2
    cpoi_x = crop_size//2

# crop the Y axis and define the position of the poi on the croped image
if poi_y < crop_size//2:
    y_start = 0
    y_end = crop_size
    cpoi_y = poi_y
elif poi_y + crop_size//2 > y_size:
    y_start = y_size - crop_size
    y_end = y_size
    cpoi_y = crop_size - (y_size - poi_y)
else:
    y_start = poi_y - crop_size//2
    y_end = poi_y + crop_size//2
    cpoi_y = crop_size//2

# Build list of data files
for i in range(delta.days + 1):
    date = d1 + datetime.timedelta(days=i)
    for hour in [0, 6, 12, 18]:
        f = (
            ANL_DATA + 'gfsanl_4_%04d%02d%02d_%02d00_%03d.grb2'
            % (date.year, date.month, date.day, hour, 0))
        grib_list.append(f)

full_dataset = []
for x, y in zip(grib_list[:-2], grib_list[1:]):
    if os.path.exists(x) and os.path.exists(y):
        full_dataset.append((x, y))

print('full dataset size : %d' % len(full_dataset))

training_size = int(len(full_dataset) * training_ratio)
val_size = len(full_dataset) - training_size
train_paths = full_dataset[:training_size]
val_paths = full_dataset[training_size:]
print('num of training samples %d' % len(train_paths))

# make sure we don't go off index
if val_size < VAL_SAMPLES:
    VAL_SAMPLES = val_size
if training_size < TRAIN_SAMPLES:
    TRAIN_SAMPLES = training_size

print('training ratio: %s %%' % training_ratio)
print('VAL_SAMPLES: %d' % VAL_SAMPLES)
print('TRAIN SAMPLES : %d' % TRAIN_SAMPLES)

X_train = np.zeros((TRAIN_SAMPLES, 1, crop_size, crop_size), dtype=np.float)
Y_train = np.zeros(TRAIN_SAMPLES, dtype=np.float)
for i, (x_path, y_path) in enumerate(train_paths[:TRAIN_SAMPLES]):
    print('reading training file id %d' % i)
    X_grb = pygrib.open(x_path)
    X_slice = X_grb.select(name=FEATURE)[
            ANL_LEVEL]['values'][x_start:x_end, y_start:y_end]
    X_train[i][0] = X_slice
    Y_grb = pygrib.open(y_path)
    Y_slice = Y_grb.select(name=FEATURE)[
            ANL_LEVEL]['values'][x_start:x_end, y_start:y_end]
    Y_train[i] = Y_slice[cpoi_x, cpoi_y]

X_val = np.zeros((VAL_SAMPLES, 1, crop_size, crop_size), dtype=np.float)
Y_val = np.zeros(VAL_SAMPLES, dtype=np.float)
for i, (x_path, y_path) in enumerate(val_paths[:VAL_SAMPLES]):
    print('reading validation file id %d' % i)
    X_grb = pygrib.open(x_path)
    X_slice = X_grb.select(name=FEATURE)[
            ANL_LEVEL]['values'][x_start:x_end, y_start:y_end]
    X_val[i][0] = X_slice
    Y_grb = pygrib.open(y_path)
    Y_slice = Y_grb.select(name=FEATURE)[
            ANL_LEVEL]['values'][x_start:x_end, y_start:y_end]
    Y_val[i] = Y_slice[cpoi_x, cpoi_y]

full_path = os.getcwd() + dest[1:]
print('writing dataset to disk at %s...' % full_path)
with h5py.File(dest, 'w') as f:
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('X_val', data=X_val)
    f.create_dataset('Y_train', data=Y_train)
    f.create_dataset('Y_val', data=Y_val)
print('done.')
