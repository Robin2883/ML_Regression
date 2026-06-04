import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

#loaded the dataset
df=pd.read_csv('Position_Salaries.csv')
df.head()

df.info()
df.describe()
df.isnull().sum()

#separating independent & dependent variables
X=df.iloc[:, 1:-1].values
y=df.iloc[:, -1].values
print(X)
print(y)

#check linearity
plt.scatter(X,y)

#splitting dataset
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)

#training model
regr=DecisionTreeRegressor(random_state=42)
regr.fit(X_train,y_train)

#predicting with model
y_pred=regr.predict(X_test)
print(y_pred)

residual=y_test-y_pred
plt.plot(residual)


sns.kdeplot(residual, color='red')

#checking homoscedasticity
plt.scatter(y_pred, residual, color='red')
plt.scatter(y_test, residual, color='blue')

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(X.min(), X.max(), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regr.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.savefig('decision_tree_reg.png')
plt.show()

#evaluation metrics
score= r2_score(y_test, y_pred)
print(f"R2 Score: {score}")
rmse=root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse}")