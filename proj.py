import pandas as pd
from numpy import *
import numpy as np
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn import metrics
import pickle

from sklearn.model_selection import train_test_split


data =pd.read_csv('train dataset.csv')
array = data.values

for i in range(len(array)):
	if array[i][0]=="Male":
		array[i][0]=1
	else:
		array[i][0]=0


df=pd.DataFrame(array)
#########defining features 
maindf =df[[0,1,2,3,4,5,6]]
mainarray=maindf.values
#print (mainarray)

########defining target of training set
temp=df[7]
train_y =temp.values
# print(mainarray)




mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
mul_lr.fit(mainarray, train_y)

testdata =pd.read_csv('test dataset.csv')
test = testdata.values

for i in range(len(test)):
	if test[i][0]=="Male":
		test[i][0]=1
	else:
		test[i][0]=0

##############test data frame##################
df1=pd.DataFrame(test)
###### test feature ######
testdf =df1[[0,1,2,3,4,5,6]]
maintestarray=testdf.values
#print(maintestarray)

y_pred = mul_lr.predict(maintestarray)
#print(y_pred)
# Saving model to disk
pickle.dump(mul_lr , open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 1,2,3,4,5,8]]))

