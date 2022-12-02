# image-classifiers README
## Background
To classify traffic sign images in ppm, svm, random forest and naive Bayes classifiers will be used.

## Requirements
matplotlib, csv, PIL, numpy, sklearn, time, datetime, pickle
```sh
import matplotlib.pyplot as plt
import csv
from PIL import Image
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import time
from datetime import timedelta
```

## Usage 
Put all the file into one folder to run the project.
If you want to change the pathway of reading files, find your local corresponding pathway of file storage
```sh
testImages, testLabels = readTrafficSigns('TrafficSignData/Training')
```

## Maintainers
[@asapacsin](https://github.com/asapacsin).
[@Rosamondrosa](https://github.com/Rosamondrosa).
