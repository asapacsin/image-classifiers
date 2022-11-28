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
            img=img.resize((32,32), Image.BICUBIC)
            img=np.array(img) #convert to gray scale
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

#train a SVM
lin_clf = svm.LinearSVC()
start = time.time()
lin_clf.fit(X, Y)
end = time.time()


# predict over training data 
Ypred=lin_clf.predict(X)

#cross validation using Randomforest
#from sklearn.ensemble import RandomForestClassifier
#accuracy_score = []
#class_num = []
#find the best number of classifier
#start = time.time()
#for i in range(2,40):
    #print(f'currently training using {i}classifiers')
    #clf=RandomForestClassifier(n_estimators=i)
    #score = cross_val_score(clf,X,Y,cv=5)
    #accuracy_score.append(score.mean())
    #class_num.append(i)
#end = time.time()
#time_cost = int(end-start)
#time_cost = timedelta(seconds = time_cost)
#print('the training model time cost is:'+str(time_cost))
#plt.scatter(class_num,accuracy_score)
#f = np.polyfit(class_num,accuracy_score,2)
#yvals = np.polyval(f,class_num)
#plot1 = plt.plot(class_num,accuracy_score,'s',label='original values')
#plot2 = plt.plot(class_num,yvals,'r',label='polyfit values')
#plt.xlabel('class nums')
#plt.ylabel('score')
#plt.legend(loc=4)
#plt.title('polyfitting')
#plt.show()

#train model using random forest
#from sklearn.ensemble import RandomForestClassifier
#clf=RandomForestClassifier(n_estimators=26)
#start = time.time()
#clf.fit(X,Y)
#end = time.time()

#print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
time_cost = int(end-start)
time_cost = timedelta(seconds = time_cost)
print('the training model time cost is:'+str(time_cost))
#Ypred=clf.predict(X)

#check the accuracy
print(accuracy_score(Y,Ypred))


# save model
import pickle
pickle.dump(lin_clf,open('model/svm_linear.sav','wb'))



