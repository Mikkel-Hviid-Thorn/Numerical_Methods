# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 20:04:28 2022

@author: Mikkel Hviid Thorn

This file contains a gradient descent method based upon: 
    Steepest descent, or descent direction equal to the negative gradient.
    Constant step or approximative line search by backtracking.
    Numeric approximation of the gradient or exact gradient given.

The algorithm searches for a critical point. The optimization problem should be unconstrained.

Be aware of the limits of the program such as the restriction on iterations.
"""


import numpy as np


# Utility Functions

def inner_product(x, y):
    """
    Takes the Euclidean inner product between to real valued vectors.
    """
    if isinstance(x, float): # For numbers.
        return x * y
    else: # For vectors with more than one coordinate.
        return sum([x[i] * y[i] for i in range(len(x))])

def norm(x):
    """
    Takes the Euclidean norm of a real valued vector.
    """
    return (inner_product(x, x)) ** (1 / 2)


# Numeric Approximation of a Gradient

def derivative(f, x, epsilon = 10 ** (-4), rate = 1 / 2):
    """
    Numeric Approximation of a derivative for the function f at x.

    Parameters
    ----------
    f : function
        A real valued differentiable function.
    x : float
        The point of differentiation.
    epsilon : float
        Treshhold for acceptable approximation. The default is 10**(-4).
    rate : float
        Rate of decreasment in difference of input values in each iteration. The default is 1/2.

    Returns
    -------
    TYPE : float
        A numeric approximation of the derivative of f at x.
    """
    
    # Initiating the algorithm with start difference h, and two approximations of the derivative.
    h = rate ** 2
    delta_f_current = (f(x + h) - f(x)) / h
    delta_f_next = (f(x + rate * h) - f(x)) / (rate * h)
    n = 1
        
    # Iterating the algorithm until convergence criteria is met or maximal allowed iterations are reached.
    while norm(delta_f_current - delta_f_next) > epsilon and n < 100:
        h *= rate
        delta_f_current = (f(x + h) - f(x)) / h
        delta_f_next = (f(x + rate * h) - f(x)) / (rate * h)
        n += 1
            
    return delta_f_next # Returns the approximation of the derivative.


def gradient(f, x, epsilon = 10 ** (-4), rate = 1 / 2):
    """
    Numeric Approximation of a gradient for the function f at x.

    Parameters
    ----------
    f : function
        A real valued differentiable function.
    x : vector
        The point of differentiation.
    epsilon : float
        Treshhold for acceptable approximation. The default is 10**(-4).
    rate : float
        Rate of decreasment in difference of input values in each iteration. The default is 1/2.

    Returns
    -------
    TYPE : vector
        A numeric approximation of the gradient of f at x.
    """
    
    def partial_derivative(i): # Defining a function to approximate the ith partial derivative.
    
        # Initiating the algorithm with start difference h, direction e_i, and two approximations of the gradient.
        h = rate ** 2
        e_i = np.zeros(len(x))
        e_i[i] = 1
        delta_f_current = (f(x + h * e_i) - f(x)) / h
        delta_f_next = (f(x + rate * h * e_i) - f(x)) / (rate * h)
        n = 1
        
        # Iterating the algorithm until convergence criteria is met or maximal allowed iterations are reached.
        while norm(delta_f_current - delta_f_next) > epsilon and n < 100:
            h *= rate
            delta_f_current = (f(x + h * e_i) - f(x)) / h
            delta_f_next = (f(x + rate * h * e_i) - f(x)) / (rate * h)
            n += 1
            
        return delta_f_next # Returns the approximation of the ith partial derivative.
    
    # Defining and returning the partial derivative.
    grad = [partial_derivative(i) for i in range(len(x))]
    return np.array(grad)



# The Steepest Descent Gradient Method

def steepest_descent_gradient_method(f, x0, gradient_function = gradient, line_method = 0, step = 1 / 10, rate = 1 / 2, epsilon = 10 ** (-2), armijo_factor = 2 / 3):
    """
    A steepest descent gradient method for finding critical points in search of a minimum of the function f.

    Parameters
    ----------
    f : function
        A real valued differentiable function.
    x0 : vector
        Initial point of search.
    gradient_function : function
        The gradient of the function f or an approximation. The default is 'gradient'.
    line_method : string or int
        The line search method used. Note that the string or int must be allowable. The default is 0 or 'constant'.
    step : float
        The initial step size at each iteration. The default is 1/10.
    rate : float
         Rate of decreasment in backtracking methods. The default is 1/2.
    epsilon : float
        Treshhold for acceptable approximation. The default is 10**(-2).
    armijo_factor : float
        A real number between zero and one used as scalar in Armijo's search method. The default is 2/3.

    Returns
    -------
    TYPE : vector
        An approximative critical point if the method was succesfull.
    """
    
    # Translating the line_method from string to integers.
    if line_method == 'constant':
        line_method = 0
    elif line_method == 'backtracking':
        line_method = 1
    elif line_method == 'armijo':
        line_method = 2
    
    def backtracking(point): # Simple direct backtracking method.
    
        # Initial step, iteration and measure of decrease.
        m = 1
        step_temp = step
        decrease = f(point - step_temp * grad) - f(point)
        
        # Iterating the algorithm until a sufficient decrease is met or maximal iterations are reached.
        while decrease >= 0 and m < 10:
            m += 1
            step_temp *= rate
            decrease = f(point - step_temp * grad) - f(point)
            
        return step_temp # Returns the found step length.
    
    def armijo(point): # Simple backtracking method using Armijo's rule.
        
        # Initial step, iteration and measure of decrease.
        m = 1
        step_temp = step
        coef = [f(point), -armijo_factor * norm(grad)]
        decrease_measure = lambda step: coef[0] + coef[1] * step - f(point - step * grad)
       
        # Iterating the algorithm until a sufficient decrease is met or maximal iterations are reached.
        while decrease_measure(step_temp) < 0 and m < 10:
            m += 1
            step_temp *= rate
            
        return step_temp # Returns the found step length.
    
    # Initializing the descent algorithm with the first step from the candidate x0.
    grad = gradient(f, x0)
    point_current = x0
    
    if line_method == 0: # Using the given line search method to find the next point.
        point_next = x0 - step * grad
    elif line_method == 1:
        point_next = x0 - backtracking(x0) * grad
    elif line_method == 2:
        point_next = x0 - armijo(x0) * grad
    
    n = 1
    
    # Iterating the algorithm until the convergence criteria is met or maximal iterations are reached.
    while norm(gradient(f, point_next)) > epsilon and n < 1000:
        grad = gradient(f, point_next)
        point_current = point_next
        
        if line_method == 0: # Using the given line search method to find the next point.
            point_next = point_current - step * grad
        elif line_method == 1:
            point_next = point_current - backtracking(point_current) * grad
        elif line_method == 2:
            point_next = point_current - armijo(x0) * grad
        
        n += 1
    
    return point_next # Returns the approximative critical point.
    

# Examples

f = lambda x: sum([x[i] ** 2 for i in range(3)])
x0 = np.array([1, 1, 1])

print(steepest_descent_gradient_method(f, x0))