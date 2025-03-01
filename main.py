import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_error_for_line_given_point(b,w,points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y-(w*x+b))**2
    return totalError / float(len(points))

def step_gradient(b_current,w_current,points,learning_rate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient = -(2/N)*(y-((w_current*x)+b_current))
        w_gradient = -(2/N)*x*(y-((w_current*x) + b_current))
    new_b = b_current - learning_rate*b_gradient
    new_w = w_current - learning_rate*w_gradient
    return [new_b, new_w]

def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, m = step_gradient(b, w, np.array(points), learning_rate)
    return [b,m]


n = 1000
x = np.arange(-n/2,n/2,1,dtype=np.float64)

m = np.random.uniform(0.3,0.5,(n,))
b = np.random.uniform(5,10,(n,))

y = x*m +b
print(x)
print(y)
points = np.stack((x,y), axis=1)
learning_rate = 0.0001
initial_b = 0
initial_w = 0
num_iterations = 100000
print(f"Starting gradient descent at b = {initial_b} w = {initial_w} error = {compute_error_for_line_given_point(initial_b, initial_w, points)}")
print("Running")

[b, m]= gradient_descent_runner(points , initial_b , initial_w , learning_rate , num_iterations)
print(f"Starting gradient descent at b = {b} w = {m} error = {compute_error_for_line_given_point(b, m, points)}")
