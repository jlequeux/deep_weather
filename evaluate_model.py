import numpy as np


def evaluate_model_accuracy(model, data, tolerance):

    X_train = data['X_train'][...]
    X_test = data['X_val'][...]
    Y_test = data['Y_val'][...]

    mean = np.mean(X_train)
    std = 3*np.std(X_train)

    X_test -= mean
    X_test /= std
    Y_test -= mean
    Y_test /= std

    predictions = model.predict(X_test, batch_size=10, verbose=1)

    Y_test_init = Y_test*std
    Y_test_init += mean
    predictions_init = predictions*std
    predictions_init += mean

    predict_ok = 0
    predict_ko = 0

    for i, j in zip(predictions_init, Y_test_init):
        delta = abs(j-i)
        if delta < tolerance:
            predict_ok += 1
        else:
            predict_ko += 1

    accuracy = float(predict_ok) / (predict_ok + predict_ko)
    print "accuracy : %f %%" % (accuracy*100)

    return accuracy
