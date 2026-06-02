import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, root_mean_squared_error

#loading dataset
df=pd.read_csv("50_Startups.csv")
df.head()

#checking null values
df.isnull().sum()

#checking O metrics
print((df == 0).sum())

#filled 0 metrics with median value
df[["R&D Spend", "Marketing Spend"]] = df[["R&D Spend", "Marketing Spend"]].replace(0, np.nan)
impute=SimpleImputer(missing_values=np.nan, strategy='median')
df[["R&D Spend", "Marketing Spend"]] = impute.fit_transform(df[["R&D Spend", "Marketing Spend"]] )
print((df == 0).sum())

#separating independent & dependent variables
X=df.iloc[:, :-1].values
y=df.iloc[:, -1].values
print(X)    
print(y)

#encoding categtorical data
ct=ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder='passthrough')
X=np.array(ct.fit_transform(X))
print(X)

#separate processed data
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)

#scaling the numerical features
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test) #never fit_transform test data. Otherwise there will be data leakage
print(X_train)
print(X_test)

#training the model
mult_lin_reg=LinearRegression()
mult_lin_reg.fit(X_train, y_train)

#predicting the output
y_pred=mult_lin_reg.predict(X_test)
print(y_pred)

#checking normal residual
residual=y_test-y_pred
sns.kdeplot(residual, color='red')

#checking homoscedasticity
plt.scatter(y_pred, residual, color='red')
plt.scatter(y_test, residual, color='blue')

#checking auto-correlation
plt.plot(residual)

r2_score=r2_score(y_test, y_pred)
print(f"r2 score : {r2_score}")
rmse= root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse}")


