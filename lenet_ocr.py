# USAGE
# pyconda lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# pyconda lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
from lenet import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import csv
import datetime as dt
import Augmenter

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)



def write_results(id_list, results):
    fileName = "/Users/Thistle/PycharmProjects/ocr_result/CNN_result_" + \
               dt.datetime.strftime(dt.datetime.now(), '%Y_%m_%d_%H_%M') + ".csv"

    print "--> File name [{0}]".format(fileName)
    result = []
    for single_res in results:
        result.append(single_res)

    with open(fileName, 'wb') as f:
        rows = zip(id_list, result)
        wtr = csv.writer(f)
        wtr.writerows([["Id","Character"]])
        wtr.writerows(rows)



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
# print("[INFO] downloading MNIST...")
# dataset = datasets.fetch_mldata("MNIST Original")

trainData, trainLabels = Augmenter.runAugment()


trainData = np.reshape(trainData, (trainData.shape[0],1,33,13))
trainData = trainData / 255.0

trainLabels= np.reshape(trainLabels,(trainLabels.shape[0],))

testData = trainData
testLabels = trainLabels

# how to use my own data?
# print ">>> Training dat --> ", trainData[0][0]
print ">>>Training data shape -->", trainData.shape
print ">>>Testing data shape -->", testData.shape
print ">>>Training labels shape -->", trainLabels.shape
print ">>>Testing labels shape -->", testLabels.shape


# ---------------------------------------------------------------------
# TODO get test results
id_list = []
test_instance_size = 0
with open("/Users/Thistle/PycharmProjects/OCR/test.csv") as f:
    for line in f:
        test_instance_size += 1

X_test = np.zeros((test_instance_size, 1, 33, 13))
counter = 0
with open("/Users/Thistle/PycharmProjects/OCR/test.csv") as f:
    for line in f:
        temp = line.split(',')
        id_list.append(temp[0])

        X_test[counter] = np.reshape(temp[9:438], (1,33,13))
        counter += 1

X_test = X_test / 255.0
# ---------------------------------------------------------------------



# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 98)
testLabels = np_utils.to_categorical(testLabels, 98)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=13, height=33, depth=1, classes=98,
	weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(trainData, trainLabels, batch_size=128, nb_epoch=100,
		verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=128, verbose=1)

	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))



# ------------------------------------------------------
# TODO how to output test data?  !!! change X_test
results = model.predict_classes(X_test, batch_size=32)
# print ">>>>>>> test result shape -->", results.shape
print "Results -->", results
write_results(id_list, results)
# ------------------------------------------------------


# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

"""
# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
	# classify the digit
	probs = model.predict(testData[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	# resize the image from a 28 x 28 image to a 96 x 96 image so we
	# can better see it
	image = (testData[i][0] * 255).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, str(prediction[0]), (5, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	# show the image and prediction
	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
		np.argmax(testLabels[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
"""


