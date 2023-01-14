import numpy as np
import matplotlib.pyplot as plt
import math

# Load our data set
x_train = np.array([1.0, 2.0]) #features
y_train = np.array([300.0, 500.0]) #target value

#Function to calculate the cost
def compute_cost(x, y, w, b):
   
    m = x.shape[0]
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost


# Function to calculate the gradient
def compute_gradient (x, y, w, b):

    m = x.shape[0]
    gradientW = 0
    gradientB = 0

    for i in range (m):
        f_wb = w * x[i] + b
        gradientW = gradientW + ((f_wb - y[i]) * x[i])
        gradientB = gradientB + (f_wb - y[i])

    totalGradientW = 1 / m * gradientW
    totalGradientB = 1 / m * gradientB
    
    return totalGradientW, totalGradientB


# Function to calculate the gradient descent
def gradient_descent(x, y, w, b, alpha, num_iters, cost_function, gradient_function): 
    for i in range (num_iters):
        w = w - alpha * gradient_function(x, y, w, b)[0]
        b = b - alpha * gradient_function(x, y, w, b)[1]

        
        if ((cost_function (x,y,w,b) < 0.000000001) or (i == 10000)):
            return w, b

    return w, b

# Putting everything together - Linear Regression Model
def model (x):
    m = gradient_descent (x_train, y_train, 0, 0, 0.33, 500, compute_cost, compute_gradient)[0]
    b = gradient_descent (x_train, y_train, 0, 0, 0.33, 500, compute_cost, compute_gradient)[1]

    y = math.ceil((m*x/1000 + b))
    print (f"The price for a house with {x} sqft is - {y} Thousand $.")


model (1000)
model (1200)
model (2000)

# Still in build
def find_alpha (x, y, w, b, num_iters, cost_function, gradient_function):

    iteration = 0

    for i in range (num_iters):
        for alpha in np.arange(0.01, 0.99, 0.05):

            print ("W: " + str(w))
            print ("B: " + str(b))
            print ("Alpha: " + str(alpha))


            w = w - alpha * gradient_function(x, y, w, b)[0]
            b = b - alpha * gradient_function(x, y, w, b)[1]

            if (cost_function (x,y,w,b) < 0.0001):
                iteration = i
                return i