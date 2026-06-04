
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
import statsmodels.api as sm

#data imported from csv file 
df=pd.read_csv('Salary_Data.csv') 
df.head()

# Extracting the Independent and Dependent Variables
X=df.iloc[:, :-1].values
y=df.iloc[:, -1].values
print(X)
print(y)

#checking linear or not
plt.scatter(X,y, color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
print(X_train)
print(X_test)

# Fitting Simple Linear Regression to the Training set
lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)

# Predicting the Test set results
y_pred=lin_reg.predict(X_test)
print(y_pred)

#checking normal residual
residual=y_test-y_pred
sns.kdeplot(residual, color='red')

#checking homoscedasticity
plt.scatter(y_pred, residual, color='red')
plt.scatter(y_test, residual, color='blue')

#checking auto-correlation
plt.plot(residual)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lin_reg.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig('trg_set_vs_reg_line.png')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, lin_reg.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig('test_set_vs_reg_line.png')
plt.show()

#checking evaluation metrics
r2score=r2_score(y_test, y_pred)
print(f"R² Score: {r2score}")
rmse=root_mean_squared_error(y_test, y_pred)
print(f"rmse value: {rmse}") 

#check p-values and confidence interval
X_train=sm.add_constant(X_train)
model=sm.OLS(y_train,X_train).fit(cov_type='HC3') #use cov_type when there is heteroscdecastiicity i.e. diff of variance of residuals
print(model.summary())


