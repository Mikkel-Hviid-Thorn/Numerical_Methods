# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:54:47 2021

@author: Gruppe 4.215:
         Andreas Galavits Olsen, Emilia Jun Nielsen, Esben Hjort Bonnerup,
         Mikkel Hviid Thorn og Silje Post Stroem.
         
Program to plot phase space portraits and approximations of the solutions to initial value problems
"""

import numpy as np
import matplotlib.pyplot as plt


# Autonomous first-order differentialequation for the functions x1, x2.
f = lambda x_vec: np.array([-3*x_vec[0] + x_vec[1], 2*x_vec[0] + 4*x_vec[1]])

# Area plottet, the limits
x1_lower, x1_upper = -5, 5;   x2_lower, x2_upper = -5, 5

# Number of vectors
x1_nvec, x2_nvec = 20, 20


def phase_space_portrait():
    """
    Plots a phase space portrait for the function f.
    
    The variabels are actually the values defined above.
    
    Parameters
    ----------
    f : the function in the differentialequation
    x1_lower, x1_upper, x2_lower, x2_upper : bounds for the plot
    x1_nvec, x2_nvec : number of columns and rows of vectors plottet
    """
    
    # The coordinatevalues of the points on each axis
    x1_values = np.linspace(x1_lower, x1_upper, x1_nvec) 
    x2_values = np.linspace(x2_lower, x2_upper, x2_nvec)

    # Matrices holding respectively the x1 and x2 coordinates
    X, Y = np.meshgrid(x1_values, x2_values)
    
    # Matrices holding respectively the derivative of the x1 and x2 at each coordinate
    U, V = f([X,Y])
    
    # Length of the vectors in U,V
    r = np.power(np.add(np.power(U,2), np.power(V,2)),0.5)
    
    # Plot of the phase space portrait
    plt.quiver(X, Y, U/r, V/r,color='firebrick', pivot='middle')# Vectors
    plt.xlabel('$x_1$');   plt.ylabel('$x_2$')                  # Labels
    plt.xlim(x1_lower-0.5, x1_upper+0.5)                        # Limits on the plot
    plt.ylim(x2_lower-0.5, x2_upper+0.5)
    plt.axhline(color='black');  plt.axvline(color='black')     # Add axis centered at origo
    

def initial_value_solution(x1_0, x2_0, step, ite, col):
    """
    Plots a solution to an initial value problem within the limits.
    The solution is a numerical solution by Eulers method.

    Parameters
    ----------
    x1_0 : initial value for the first coordinate vector
    x2_0 : initial value for the second coordinate vector
    step : stepsize
    ite : max iterations
    col : color of the solution
    """
    
    # List of the function values
    Z = [np.array([x1_0, x2_0])]
    
    i = 0
    while True: # Expanding the solution in one direction
        Z = Z + [f(Z[-1])*step + Z[-1]]
        i += 1
        
        # Stopping when the limits are reached or iterations
        if Z[-1][0] < x1_lower-0.2 or Z[-1][0] > x1_upper+0.2:
            break
        elif Z[-1][1] < x2_lower-0.2 or Z[-1][1] > x2_upper+0.2:
            break
        elif i > ite:
            break
  
    i = 0
    while True: # Expanding the solution in the other direction
        Z = [-f(Z[0])*step + Z[0]] + Z
        i += 1
        
        # Stopping when the limits are reached or iterations
        if Z[0][0] < x1_lower-0.2 or Z[0][0] > x1_upper+0.2:
            break
        elif Z[0][1] < x2_lower-0.2 or Z[0][1] > x2_upper+0.2:
            break
        elif i > ite:
            break
    
    # Transposing to get to list of coordinates indstead of one list with points
    Z = np.array(Z).transpose()
    
    # Plot of the curve (x1(t),x2(t)) and a point (x1_0,x2_0)
    plt.plot(Z[0], Z[1], color=col, label="($%g$, $%g$)" % (x1_0, x2_0))
    plt.plot(x1_0, x2_0, 'o', color=col)
    
    # Settings for the plot
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


plt.style.use('seaborn-dark')
plt.rc('axes', titlesize=10)     # Axis titles
plt.rc('axes', labelsize=10)     # Axis labels
plt.rc('xtick', labelsize=10)    # Numbers on first axis
plt.rc('ytick', labelsize=10)    # Numbers on second axis
plt.rc('legend', fontsize=10)    # Legend size
plt.rcParams['axes.facecolor'] = 'white'


plt.figure()
phase_space_portrait()

initial_value_solution(-3, 0, 0.005, 10000, 'navy')
initial_value_solution(0, 1, 0.005, 10000, 'green')

#plt.savefig('phase_space_portrait.png', dpi=1000, bbox_inches='tight')
