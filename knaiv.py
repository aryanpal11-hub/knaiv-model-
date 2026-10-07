import numpy as np
import pandas as pd
import seaborn as sn 
import matplotlib.pyplot as plt

#now loding the iris data 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X,y=load_iris(return_X_y=True)# with the healp of this code we load the dependent and idependent features in the X and y variable 
print(X)