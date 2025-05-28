import pandas as pd
data=pd.read_csv("C:/Users/91998/Downloads/clean_study_hours_vs_marks.csv")
print(data.head(10))
print(data.tail())
print(data.isnull().sum())

X=data[['StudyHours']]
y=data['Marks']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

print("R^2:",r2_score(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))

import matplotlib.pyplot as plt

plt.scatter(X_test,y_test, color='blue',label='Actual')
plt.plot(X_test,y_pred,color='red',linewidth=2,label='Predicted')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
