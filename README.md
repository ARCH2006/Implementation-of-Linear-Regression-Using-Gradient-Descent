# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ARCHANA S
RegisterNumber:  212223040019
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate = 0.1,num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta = np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):                    
        #calculate predictions
        predictions = (x).dot(theta).reshape(-1,1)
                     
        #calculate errors
        errors = (predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-= learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta

data = pd.read_csv("C:/Users/ANANDAN S/Documents/ML labs/50_Startups.csv")
data.head()
#Assuming the last column is your target variable y
x= (data.iloc[1:,:-2].values)
x1 = x.astype(float)
scaler = StandardScaler()
y =(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled = scaler.fit_transform(x1)
y1_scaled = scaler.fit_transform(y)
print(x)
print(x1_scaled)
#learn model parameters
theta = linear_regression(x1_scaled,y1_scaled)
#predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_scaled),theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"prediction value: {pre}")
```

## Output:
```
R&D Spend	Administration	Marketing Spend	State	Profit
0	165349.20	136897.80	471784.10	New York	192261.83
1	162597.70	151377.59	443898.53	California	191792.06
2	153441.51	101145.55	407934.54	Florida	191050.39
3	144372.41	118671.85	383199.62	New York	182901.99
4	142107.34	91391.77	366168.42	Florida	166187.94

[[-0.42925552]]
prediction value: [[192932.45813544]]
```



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
