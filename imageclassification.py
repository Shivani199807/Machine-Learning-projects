import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#from sklearn.datasets import fetch_mldata
#dataset=fetch_mldata('MNIST original')

from sklearn.datasets import load_digits
dataset_1=load_digits()#from this dataset we loaded the digits dataset

X=dataset_1.data
y=dataset_1.target

some_digit=X[200] #this will show the digit that is placed at 200 point in dataset
some_digit_image=some_digit.reshape(8,8)# size of dataset is of 8*8

plt.imshow(some_digit_image)#this will display that digit
plt.show

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)#here train and test split will take place which will be 75% training and 25% testing


from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier(max_depth=15)#this max depth is used to increase the accuracy
dtf.fit(X_train,y_train)

dtf.score(X_train,y_train)# here we will calculate the score i.e the accuracy

dtf.predict(X[[6,200,255,1000,660],0:64])#here this first braket is representinng the no. which are placed at the particular
                                          # point like at 200 place it showed 1 while in output after prediction also showed 1
from sklearn.tree import export_graphviz

export_graphviz(dtf,out_file="tree.dot")

import graphviz
with open("tree.dot") as f:
    dot_graph=f.read()
    graphviz.Source(dot_graph)





