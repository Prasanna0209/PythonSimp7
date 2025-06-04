
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("C:/Users/91998/Downloads/iris_45.csv")
print(data.head(10))
print(data.info())
print(data.describe())
print(data.isnull().sum())

x=data.drop(["target"], axis=1)
y=data["target"]

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
print("Features Processed")

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,train_size=0.8,random_state=50)
print("Data split successfully")

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
print("Model trained successfully")

train_accuracy=knn.score(x_train,y_train)
test_accuracy=knn.score(x_test,y_test)
print("Train Accuracy:",train_accuracy)
print("Test Accuracy:",test_accuracy)

y_predicted=knn.predict(x_test)
correct_prediction=(y_test==y_predicted)
wrong_prediction=(y_test!=y_predicted)
print("Correct Predictions:")
print(x_test[correct_prediction])
print("Wrong Predictions:")
print(x_test[wrong_prediction])

