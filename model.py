import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv(r'D:\Datasets\diabetes.csv')
#print(data.head())

#the columns Glucose,BloodPressure,SkinThickness,Insulin,BMI has value zero in it which is not realistically possible so lets fix it

data['Glucose'].replace(0,data['Glucose'].mean(),inplace=True)
data['BloodPressure'].replace(0,data['BloodPressure'].mean(),inplace=True)
data['SkinThickness'].replace(0,data['SkinThickness'].mean(),inplace=True)
data['Insulin'].replace(0,data['Insulin'].mean(),inplace=True)
data['BMI'].replace(0,data['BMI'].mean(),inplace=True)

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#data = pd.DataFrame(scaler.fit_transform(data),columns=data.columns)

x = data.drop('Outcome',axis=1)
y = data['Outcome']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=75)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

from sklearn import metrics
#print(metrics.confusion_matrix(y_test,y_pred))
print("The accuracy of model is :-",metrics.accuracy_score(y_test,y_pred))
#print(metrics.classification_report(y_test,y_pred))

print(logreg.predict([[6,148,72,35,79.79,33.6,0.627,50]]))

pickle.dump(logreg,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print("After dumping the model :-")
print(model.predict([[6,148,72,35,79.79,33.6,0.627,50]]))