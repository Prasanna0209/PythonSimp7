import pandas as pd
from fontTools.misc.macCreatorType import setMacCreatorAndType
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

from CSV import X_train, X_test

data=pd.read_csv("C:/Users/91998/Downloads/iris.csv")
print(data.head(10))
print(data.tail())
print(data.isnull().sum())
print(data.info())
print(data.describe())

X=data.drop("target",axis=1)
y=data["target"]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print("Featured preprocessed")

X_train,X_test,y_train,y_test=(train_test_split(X_scaled,y,test_size=0.2,train_size=0.8,random_state=50))
print("Data split successfully")

model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
print("Model is trained successfully")

train_score=model.score(X_train,y_train)
print("Train score is ",train_score)

new_sample=np.array([[5.2,4.8,2.6,0.8]])
new_scale=scaler.transform(new_sample)
prediction=model.predict(new_scale)
print("Prediction:",prediction)
