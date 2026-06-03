import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

#loading the dataset
df=pd.read_csv('Position_Salaries.csv')
df.head()

#defining independent and dependent variables
X=df.iloc[:, 1:-1].values
y=df.iloc[:, -1].values
print(X)
print(y)

#check linearity
plt.scatter(X,y)
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.title('Posn VS Salary')
plt.show()

# - No splitting data as dataset is very low

# Training the Linear Regression model on the whole dataset
lin_reg=LinearRegression()
lin_reg.fit(X,y)

# Training the Polynomial Regression model on the whole dataset
poly_reg=PolynomialFeatures(degree=4) #ML Engr has to find out the reqr degree. If too low: underfitting, if too high: overfitting. Balance.
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly, y)

#predicting with trained model
y_pred=lin_reg_2.predict(X_poly)
print(y_pred)
print(y)

#checkin residual value
residual=y_pred-y
print(residual)

#auto-correlation
plt.plot(residual)

#normal residual
sns.displot(residual, kde=True)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))



