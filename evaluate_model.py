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

    # both matrix needs to have the same shape
    predictions_init = predictions_init.reshape(predictions_init.shape[0])

    # calculate the purcentage of prediction within the tolerance
    accuracy = np.mean(np.abs(predictions_init - Y_test_init) < tolerance)
    print "accuracy : %f %%" % (accuracy*100)

    return accuracy


def evaluate_model_tolerance(model, data, target_accuracy):

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

    # both matrix needs to have the same shape
    predictions_init = predictions_init.reshape(predictions_init.shape[0])

    # calc tolerance so that 'target_accuracy' purcent of predictions are true
    tolerance = 999
    for i in range(1000):
        tol = float(i)/100
        if target_accuracy < np.mean(
                np.abs(predictions_init - Y_test_init) < tol):
            tolerance = tol
            break

    print "tolerance : %f " % tolerance
    return tolerance
