import numpy as np
#import as module ocr_loader. Be careful, when import as module,
# only the first time it will execute, after import it will not execute any more


def load_data():
    # import data from bitmap into array
    from numpy import genfromtxt
    training_data_all = genfromtxt('train.csv', delimiter=',')
    from numpy import genfromtxt
    test_data = genfromtxt('test.csv', delimiter=',')

    #split training data into training and validation adjusted by train_ratio
    train_ratio = 0.6

    #get size of input instance
    trainX = training_data_all.shape[0]

    #split original training data into training and validation parts
    training_data = training_data_all[:int(trainX * train_ratio):,:]
    validation_data = training_data_all[int(trainX * train_ratio):,:]

    return (training_data, validation_data, test_data)



def load_data_wrapper():
    tr_data, va_data, te_data = load_data()

    #obtain bitmap data for training and standarized [0,255) into [0,1)
    training_inputs = [np.reshape(x, (429, 1)) for x in tr_data[:,9:]/255]

    #vectorize & tr_data[:,1] is (30169,), tr_data[:,[1]] is (30169,1)
    #very important!! tr_data[:,1] need to convert to int
    training_outputs = [vectorized_result(y) for y in tr_data[:,1].astype(int)]

    training = zip(training_inputs, training_outputs)

    # obtain bitmap data for validation and standarized [0,255] into [0,1)
    # validation_inputs = va_data[:,9:]/255 - this does not work
    validation_inputs = [np.reshape(x, (429, 1)) for x in va_data[:, 9:] / 255]
    validation = zip(validation_inputs, va_data[:,[1]].astype(int))

    # obtain bitmap data for test and standarized [0,255) into [0,1)
    # test_inputs = te_data[:, 9:]/255 - this does not work
    test_inputs = [np.reshape(x, (429, 1)) for x in te_data[:, 9:] / 255]
    test = zip(test_inputs, te_data[:,[1]].astype(int))

    return (training, validation, test)


def vectorized_result(j):
    """Return a 98-dimensional unit vector with a 1 in the jth
    position and zeroes elsewhere.  This is used to convert a character
    from 98 classes into a corresponding desired output from the neural
    network."""
    e = np.zeros((98, 1),dtype=np.int8)
    e[j] = 1
    return e

# SML_OCR
