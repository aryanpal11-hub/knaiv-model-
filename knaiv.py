import numpy as np
import pandas as pd
import seaborn as sn 
import matplotlib.pyplot as plt

#now loding the iris data 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X,y=load_iris(return_X_y=True)# with the healp of this code we load the dependent and idependent features in the X and y variable 
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


#now coming to the model training part 
from sklearn.naive_bayes import GaussianNB
model1=GaussianNB()
model1.fit(x_train,y_train)


#now coming to the prediction part 
y_pred=model1.predict(x_test)


#now checking the performance of the model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))