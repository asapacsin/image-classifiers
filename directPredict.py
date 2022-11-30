import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
import pickle



#load the given image path
img = Image.open('Test/Test/directTest.ppm')
#process the image
img = img.resize((32,32),Image.BICUBIC)
img = np.array(img)
img = img.flatten()
#load the model
model = pickle.load(open('model/randomForest.sav','rb'))
#predict the image
Ypred=model.predict(img)
#output prediction
print(Ypred)