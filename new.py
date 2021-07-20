import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import *
import pickle

data = pd.read_csv('train dataset.csv')
size = data.shape

data_2 = pd.read_csv('test dataset.csv')
size2 = data_2.shape



Z = data.iloc[:,:].values
W = data_2.iloc[:,:].values


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Z[:,0]= le.fit_transform(Z[:,0])
Z[:,-1]= le.fit_transform(Z[:,-1])
W[:,0]= le.fit_transform(W[:,0])
W[:,-1]= le.fit_transform(W[:,-1])

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

# ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[7])], remainder='passthrough')
# Z = np.array(ct.fit_transform(Z))
# W = np.array(ct.fit_transform(W))

X_train = Z[:,:-1]
y_train = Z[:,-1]
y_train=y_train.astype('int')


X_test = W[:,:-1]
y_test = W[:,-1]
y_test=y_test.astype('int')




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


prediction = classifier.predict(sc.transform([[2,4,5,2,3,2,4]]))
print(prediction)

y_pred = classifier.predict(X_test)
side_by_side = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
print(side_by_side)

pickle.dump(classifier , open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 1,2,3,4,5,8]]))


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print('Accuracy=', acc)
print('Confusion Matrix=', cm)
