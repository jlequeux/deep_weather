# deep_weather
Naive deep learning pipeline for weater data.

download_weather_data.py :
    this script will download the images from noaa

load.py:
    create a hdf5 file with the images croped arround the point of interest
    by default images are stored in ./loaded_data/

setup_model.py:
    use this to modify the settings and generates several models

train_model.py:
    this script will run all the models described in "setup_model.py"
    results will be stored in a CSV file in ./results/
    a tensorboard log is also generated in ./logs/

write_result.py:
    function to use in order to write the result of the training in the CSV
