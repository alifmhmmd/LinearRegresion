import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim


# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]


# Model

a = 0.0001 #Learning rate
n = len(X) #number of elements in X
m = 0
b = 0
# Gradient Descent



for i in range(50):
    Y_pred = m*X + b # predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_b = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - a * D_m  # Update m
    b = b - a * D_b  # Update c
    axes = plt.gca()
    axes.set_xlim([20,80])
    axes.set_ylim([0,120])
    plt.scatter(X, Y,color="green",marker='+') 
    plt.plot([min(X),max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
    plt.pause(0.5)
    plt.clf()

    

plt.show()