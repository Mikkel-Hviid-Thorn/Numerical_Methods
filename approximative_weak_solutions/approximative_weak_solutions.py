import numpy as np
import scipy as sp


def inner_product(f, g, a: float, b: float):
    """
    Computes the inner product in L^2((a,b)) between f and g.
    """
    return sp.integrate.quad(lambda x: f(x)*g(x), a, b)[0]

def poisson_eigenfunction_decomposition(h, n: int):
    """
    Computes the truncated sum of the eigenfunction decomposition for the solution operator for the Poisson equation on
    (0,pi), see Example 6.5.2.
    """
    eigenvalues = [1/(j**2) for j in range(1, n+1)]
    def eigenfunction(j):
        ef = lambda x: np.sqrt(2/np.pi)*np.sin(j*x)
        return ef
    eigenfunctions = [eigenfunction(j) for j in range(1, n+1)]
    coefficients = [inner_product(h, ef, 0, np.pi) for ef in eigenfunctions]
    solution = lambda x: np.sum(np.array(eigenvalues)*np.array(coefficients)*np.array([ef(x) for ef in eigenfunctions]))
    return solution

def sturm_liouville_eigenfunction_decomposition(h, n: int):
    """
    Computes the truncated sum of the eigenfunction decomposition for the solution operator for the Sturm-Liouville
    equation from Example 6.1.4, see Example 6.5.3.
    """
    eigenvalues = [1/((np.pi*j/np.log(2))**2+1/4) for j in range(1, n+1)]
    def eigenfunction(j):
        ef = lambda x: np.sqrt(2/(np.log(2)*x))*np.sin(np.pi*j*np.log(x)/np.log(2))
        return ef
    eigenfunctions = [eigenfunction(j) for j in range(1, n+1)]
    coefficients = [inner_product(h, ef, 1, 2) for ef in eigenfunctions]
    solution = lambda x: np.sum(np.array(eigenvalues)*np.array(coefficients)*np.array([ef(x) for ef in eigenfunctions]))
    return solution

def derivative(f):
    """
    Numerical approximation to the derivative of f.
    """
    return lambda x: (f(x+1e-2)-f(x))/(1e-2)

def poisson_bilinear_form(f, g):
    """
    Defines the bilinear form associated with the weak formulation of the Poisson equation on (0,pi), see Example
    6.2.10.
    """
    df = derivative(f)
    dg = derivative(g)
    return sp.integrate.quad(lambda x: df(x)*dg(x), 0, np.pi)[0]

def sturm_liouville_bilinear_form(f, g):
    """
    Defines the bilinear form associated with the weak formulation of the Sturm-Liouville equation from Example 6.1.4,
    see Example 6.2.11.
    """
    df = derivative(f)
    dg = derivative(g)
    return sp.integrate.quad(lambda x: (x**2)*df(x)*dg(x), 1, 2)[0]

def piecewise_linear_approximation(bilinear_form, h, a: float, b: float, n: int):
    """
    Computes a piecewise linear approximate solution to a weak formulation on (a,b) with a partition consisting of n+2
    equidistant points. The associated bilinear form is bilinear_form and input function h.
    """
    def basis_function(a, b, j, n):
        bf = lambda x: 0 if x<a+(j-1)*(b-a)/(n+1) else 1+(n+1)/(b-a)*(x-a-j*(b-a)/(n+1)) if x<a+j*(b-a)/(n+1) else 1-(n+1)/(b-a)*(x-a-j*(b-a)/(n+1)) if x<a+(j+1)*(b-a)/(n+1) else 0
        return bf
    basis = [basis_function(a, b, j, n) for j in range(1, n+1)]
    vector_coefficients = [inner_product(h, bf, a, b) for bf in basis]
    matrix_coefficients = [[bilinear_form(bf_col, bf_row) for bf_col in basis] for bf_row in basis]
    coefficients = sp.linalg.solve(np.array(matrix_coefficients), np.array(vector_coefficients))
    solution = lambda x: np.sum(coefficients*np.array([bf(x) for bf in basis]))
    return solution

def plot_solution(equation, method, h, n):
    """
    Plot approximative solutions with specified equation, method, input function h, and order n.
    """
    if equation == 'Poisson' and method == 'eigenfunction decomposition':
        solution = poisson_eigenfunction_decomposition(h, n)
        interval = np.linspace(0, np.pi, 100)
    elif equation == 'Poisson' and method == 'piecewise linear':
        solution = piecewise_linear_approximation(poisson_bilinear_form, h, 0, np.pi, n)
        interval = np.linspace(0, np.pi, 100)
    elif equation == 'Sturm-Liouville' and method == 'eigenfunction decomposition':
        solution = sturm_liouville_eigenfunction_decomposition(h, n)
        interval = np.linspace(1, 2, 100)
    elif equation == 'Sturm-Liouville' and method == 'piecewise linear':
        solution = piecewise_linear_approximation(sturm_liouville_bilinear_form, h, 1, 2, n)
        interval = np.linspace(1, 2, 100)
    else:
        print('Error, equation {} or method {} is not supported'.format(equation, method))

    return [solution(x) for x in interval]


import matplotlib.pyplot as plt

plt.style.use('seaborn-dark')
plt.rc('axes', titlesize=10)     # Axis titles
plt.rc('axes', labelsize=10)     # Axis labels
plt.rc('xtick', labelsize=10)    # Numbers on first axis
plt.rc('ytick', labelsize=10)    # Numbers on second axis
plt.rc('legend', fontsize=10)    # Legend size
plt.rcParams['axes.facecolor'] = 'white'

def example_Poisson(h, file_id):
    interval = np.linspace(0, np.pi, 100)

    plt.figure()
    solution = plot_solution('Poisson', 'eigenfunction decomposition', h, 2)
    plt.plot(interval, solution, color='firebrick', label='2')
    solution = plot_solution('Poisson', 'eigenfunction decomposition', h, 4)
    plt.plot(interval, solution, color='navy', label='4')
    solution = plot_solution('Poisson', 'eigenfunction decomposition', h, 6)
    plt.plot(interval, solution, color='forestgreen', label='6')
    plt.legend(bbox_to_anchor=(1.02, 1), title='Terms', loc='upper left')
    plt.savefig('poisson_eigenfunction_decomposition_{}.png'.format(file_id), dpi=1000, bbox_inches='tight')

    plt.figure()
    solution = plot_solution('Poisson', 'piecewise linear', h, 2)
    plt.plot(interval, solution, color='firebrick', label='4')
    solution = plot_solution('Poisson', 'piecewise linear', h, 4)
    plt.plot(interval, solution, color='navy', label='6')
    solution = plot_solution('Poisson', 'piecewise linear', h, 6)
    plt.plot(interval, solution, color='forestgreen', label='8')
    plt.legend(bbox_to_anchor=(1.02, 1), title='Partition', loc='upper left')
    plt.savefig('poisson_piecewise_linear_{}.png'.format(file_id), dpi=1000, bbox_inches='tight')

def example_Sturm_Liouville(h, file_id):
    interval = np.linspace(1, 2, 100)

    plt.figure()
    solution = plot_solution('Sturm-Liouville', 'eigenfunction decomposition', h, 2)
    plt.plot(interval, solution, color='firebrick', label='2')
    solution = plot_solution('Sturm-Liouville', 'eigenfunction decomposition', h, 4)
    plt.plot(interval, solution, color='navy', label='4')
    solution = plot_solution('Sturm-Liouville', 'eigenfunction decomposition', h, 6)
    plt.plot(interval, solution, color='forestgreen', label='6')
    plt.legend(bbox_to_anchor=(1.02, 1), title='Terms', loc='upper left')
    plt.savefig('sturm_liouville_eigenfunction_decomposition_{}.png'.format(file_id), dpi=1000, bbox_inches='tight')

    plt.figure()
    solution = plot_solution('Sturm-Liouville', 'piecewise linear', h, 2)
    plt.plot(interval, solution, color='firebrick', label='4')
    solution = plot_solution('Sturm-Liouville', 'piecewise linear', h, 4)
    plt.plot(interval, solution, color='navy', label='6')
    solution = plot_solution('Sturm-Liouville', 'piecewise linear', h, 6)
    plt.plot(interval, solution, color='forestgreen', label='8')
    plt.legend(bbox_to_anchor=(1.02, 1), title='Partition', loc='upper left')
    plt.savefig('sturm_liouville_piecewise_linear_{}.png'.format(file_id), dpi=1000, bbox_inches='tight')


if __name__ == "__main__":
    example_functions_poisson = [[lambda x: x, 'id'],
                                 [lambda x: x**3-3*x**2-x+3, 'poly'],
                                 [np.exp, 'exp'],
                                 [lambda x: np.sqrt(2/np.pi)*np.sin(x), 'ef'],
                                 [lambda x: (x>np.pi/3)*1-(x>2*np.pi/3)*1, 'indicator']]

    for exmp in example_functions_poisson:
        example_Poisson(exmp[0], exmp[1])

    example_functions_sturm_liouville = [[lambda x: x, 'id'],
                                         [lambda x: x**3-3*x**2-x+3, 'poly'],
                                         [np.exp, 'exp'],
                                         [lambda x: np.sqrt(2/(np.log(2)*x))*np.sin(np.pi*np.log(x)/np.log(2)), 'ef'],
                                         [lambda x: (x>4/3)*1-(x>5/3)*1, 'indicator']]

    for exmp in example_functions_sturm_liouville:
        example_Sturm_Liouville(exmp[0], exmp[1])
        