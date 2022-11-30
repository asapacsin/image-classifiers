from sklearn.metrics import accuracy_score
import pickle


# predict over training data 
model = pickle.load(open('model/randomForest.sav','rb'))


#check the accuracy
print(accuracy_score(Y,Ypred))