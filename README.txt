In the terminal, cd to target directory: 

>>>python2

>>>import ocr_loader
>>>training_data, validation_data, test_data = ocr_loader.load_data_wrapper()

>>>import network2
>>>net = network2.Network([429, 100, 98], cost=network2.CrossEntropyCost)
>>>net.SGD(training_data, 30, 10, 3.0, lmbda = 0.0, evaluation_data=validation_data,\
...monitor_evaluation_cost=True,monitor_evaluation_accuracy=True,monitor_training_cost= \
...True,monitor_training_accuracy=True)# SML_OCR
# SML_OCR
