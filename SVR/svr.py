import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('Position_Salaries.csv')
df.head()

df.info()
df.describe()
df.isnull().sum()

#separating independent & dependent variables
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
y=y.reshape(-1,1)
print(X)
print(y)

sc_y = StandardScaler()
y_scaled = sc_y.fit_transform(y).ravel()
print(y_scaled)

#check linearity
plt.scatter(X,y)
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.title('Posn VS Salary')
plt.show()

#use it if dataset is pretty simple and already in scaled form
#reg=SVR(kernel='rbf') #'rbf' is used for non-linear regression, otherwise use 'linear'
#reg.fit(X,y)

#y_pred=reg.predict(X)
#print(y_pred)

# # PIPELINE
# - In this way, X gets automatically scaled, but not y. That's why we scaled y beforehand.
# - Another imp pt is that, **StandardScaler** takes **2D** array. Later it's converted again to **1D** array for **SVR** input.

#we will follow this way for real-world applications

regr = make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.02))  #tune C and epsilon to get good R2_Score
regr.fit(X, y_scaled)
#Pipeline(steps=[('standardscaler', StandardScaler()), ('svr', SVR(epsilon=0.2))])

y_pred_scaled = regr.predict(X)
print(y_pred_scaled)

y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1,1))
print(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, sc_y.inverse_transform(y_pred_scaled.reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.savefig('svr.png')
plt.show()

residual=y-y_pred
print(residual)
sns.kdeplot(residual, color='red')

#checking homoscedasticity
plt.scatter(y_pred, residual, color='red')
plt.scatter(y, residual, color='blue')

#checking auto-correlation
plt.plot(residual)

#evaluation metrics
score=r2_score(y, y_pred)
print(f"r2 score : {score}")
rmse= root_mean_squared_error(y, y_pred)
print(f"RMSE: {rmse}")


