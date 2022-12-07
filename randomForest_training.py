import matplotlib.pyplot as plt
import csv
from PIL import Image
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import time
from datetime import timedelta

def readTrafficSigns(rootpath):
    '''Reads traffic sign data 
    Arguments: path to the traffic sign data, for example './TrafficSignData/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over N classes, at most we have 42 classes
    N=15
    for c in range(0,N):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        #gtReader.next() # skip header
        next(gtReader)
        # loop over all images in current annotations file
        for row in gtReader:
            img=Image.open(prefix + row[0])  # the 1th column is the filename
            # preprocesing image, make sure the images are in the same size
            img=img.resize((32,32), Image.BICUBIC).convert('L') #convert to gray scale
            img=np.array(img) 
            images.append(img) 
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

# load the images
trainImages, trainLabels = readTrafficSigns('TrafficSignData/Training')
# print number of historical images
print('number of historical data=', len(trainLabels))
# show one sample image
#plt.imshow(trainImages[44])
#plt.show()

# design the input and output for model
X=[]
Y=[]
for i in range(0,len(trainLabels)):
    # input X just the flattern image, you can design other features to represent a image
    X.append(trainImages[i].flatten())
    Y.append(int(trainLabels[i]))
X=np.array(X)
Y=np.array(Y)

#train a random forest
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=60)
start = time.time()
clf.fit(X,Y)
end = time.time()
time_cost = timedelta(seconds=int(end - start))
print('time cost:', time_cost)
Ypred = clf.predict(X)

#check the accuracy
print(accuracy_score(Y,Ypred))


# save model
import pickle
pickle.dump(clf,open('model/randomForest.sav','wb'))



