{\rtf1\ansi\ansicpg936\cocoartf1404\cocoasubrtf340
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red53\green53\blue53;\red83\green83\blue83;\red83\green83\blue83;
}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab560
\pard\pardeftab560\slleading20\partightenfactor0

\f0\b\fs24 \cf2 In the terminal, cd to target directory: \
\
\pard\pardeftab720\sl408\partightenfactor0

\f1\b0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 >>>
\f0\b \cf2 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 python2\
\pard\pardeftab560\slleading20\partightenfactor0
\cf2 \
\pard\pardeftab720\sl408\partightenfactor0

\f1\b0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 >>>
\f0\b \cf2 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 import
\b0  
\b ocr_loader\

\f1\b0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 >>>
\f0 \cf2 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 training_data, validation_data, test_data = ocr_loader.load_data_wrapper()\
\
\pard\pardeftab720\sl408\partightenfactor0

\f1 \cf4 \expnd0\expndtw0\kerning0
>>>
\f0\b \cf2 \kerning1\expnd0\expndtw0 import
\b0  
\b network2\
\pard\pardeftab720\sl408\partightenfactor0

\f1\b0 \cf4 \expnd0\expndtw0\kerning0
>>>
\f0 \cf2 \kerning1\expnd0\expndtw0 net = network2.Network([429, 100, 98], cost=network2.CrossEntropyCost)\

\f1 \cf4 \expnd0\expndtw0\kerning0
>>>
\f0 \cf2 \kerning1\expnd0\expndtw0 net.SGD(training_data, 30, 10, 3.0, lmbda = 0.0, evaluation_data=validation_data,\\\
\pard\pardeftab720\sl408\partightenfactor0

\f1 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 ...
\f0 \cf2 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 monitor_evaluation_cost=True,monitor_evaluation_accuracy=True,monitor_training_cost= \\\
\pard\pardeftab720\sl408\partightenfactor0

\f1 \cf4 \expnd0\expndtw0\kerning0
...
\f0 \cf2 \kerning1\expnd0\expndtw0 True,monitor_training_accuracy=True)
\b \

\b0 \
\pard\pardeftab560\slleading20\partightenfactor0
\cf2 \
}# SML_OCR
# SML_OCR
