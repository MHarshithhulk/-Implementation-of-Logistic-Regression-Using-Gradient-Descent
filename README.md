# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: GAUTHAM KRISHNA S
RegisterNumber:  212223240036
*/
```
```PY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
Y

theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta

theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(y_pred)

print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

## Output:

### Dataset
<img width="1093" alt="Screenshot 2024-04-23 at 9 23 23 AM" src="https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/58af8fc6-877e-4c18-881c-c886b50e8368">

### Dataset.dtyes
<img width="1093" alt="Screenshot 2024-04-23 at 9 23 50 AM" src="https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/a2c10f53-9f25-4873-bdac-dbb0518454a6">

### Datset cat.codes
<img width="1101" alt="Screenshot 2024-04-23 at 9 24 11 AM" src="https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/e3cd985c-dbfa-497e-8dfd-7d9346601861">

### Y
<img width="1101" alt="Screenshot 2024-04-23 at 9 24 31 AM" src="https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/ad8b7cd1-d811-4655-97d4-708248fd5206">

### Accuracy
<img width="1101" alt="Screenshot 2024-04-23 at 9 26 16 AM" src="https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/9d49c71e-ce5a-48bd-9ae9-aee5e2b658e8">

### Y_pred
<img width="1101" alt="Screenshot 2024-04-23 at 9 26 30 AM" src="https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/7f2bcfec-cbac-4a64-927c-cba98033ed1d">

### Y
<img width="1101" alt="Screenshot 2024-04-23 at 9 26 43 AM" src="https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/ddbcc6fa-1577-4f34-8347-c9a384fb81d3">

### Y_prednew for input 1
<img width="1101" alt="Screenshot 2024-04-23 at 9 26 57 AM" src="https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/bfc24a6c-03e7-4dbb-b67b-166422e4719a">

### Y_prednew for input 2
<img width="1101" alt="Screenshot 2024-04-23 at 9 26 57 AM" src="https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/50efa5d7-e17a-49d1-9018-677aad6b2067">


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

