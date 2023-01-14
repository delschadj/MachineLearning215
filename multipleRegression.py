import numpy as np
import matplotlib.pyplot as plt
import copy, math

# Load our data set
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]) #features
y_train = np.array([460, 232, 178]) #target value

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict (x, w, b):
    p = np.dot (x, w) + b

    return p

#Function to calculate the cost
def compute_cost(X, y, w, b):

    # X = Data-Matrix
    # m examples with n features
   
    m = X.shape[0]
    cost = 0.0
    
    for i in range(m):
        f_wb = np.dot (X[i], w) + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

print (compute_cost (X_train , y_train , w_init , b_init))

# Function to calculate the gradient
def compute_gradient (X, y, w, b):

    m,n = X.shape
    gradientW = np.zeros((n,))
    gradientB = 0

    for i in range (m):
        f_wb = np.dot (X[i], w) + b - y[i]

        for j in range (n):
            gradientW[j] = (f_wb) * X[i][j]
        
        gradientB = gradientB + (f_wb - y[i])
        

    totalGradientW = 1 / m * gradientW
    totalGradientB = 1 / m * gradientB
    
    return totalGradientW, totalGradientB

# Function to calculate the gradient descent
def gradient_descent(X, y, initalW, initalB, alpha, num_iters, cost_function, gradient_function): 
    
    w = copy.deepcopy(initalW)
    b = initalB

    for i in range (num_iters):
        w = w - alpha * gradient_function(X, y, w, b)[0]
        b = b - alpha * gradient_function(X, y, w, b)[1]

        
        if ((cost_function (X,y,w,b).any() < 0.000000001) or (i == 10000)):
            return w, b

    return w, b



# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, iterations,
                                                    compute_cost, compute_gradient, )
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")















# Putting everything together - Linear Regression Model
def model (x):
    initial_w = np.zeros_like(w_init)
    initial_b = 0.0

    # some gradient descent settings
    iterations = 1000
    alpha = 5.0e-7

    w_final, b = gradient_descent (X_train, y_train, initial_w, initial_b, alpha, iterations, compute_cost, compute_gradient)

    y = np.dot (x, w_final) + b
    print (f"The price for a house with {x} sqft is - {y} Thousand $.")