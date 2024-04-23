"""
Implement the conjugate gradient (refined version) algorithm and use to it solve linear systems in which A is the Hilbert 
matrix, whose elements are
                                    A(i,j) = 1 / (i + j − 1)
Set the right-hand-side to b^T = (1, 1, . . . , 1), and the initial point to x0 = 0. Try dimensions n = 5, 8, 12, 20 and 
report the number of iterations required to reduce the residual below 10^(−7).
"""

import numpy as np
import numpy.linalg as la

# Conjugate gradient algorithm (refined version)
def conjugate_gradient(A, b, x0, tolerence):
    r = np.dot(A, x0) - b  # Initial residual of system Ax = b 
    p = -r                 # Initial conjugate direction
    iterations = 0         # Iteration count
    x = x0                 # Initial approximation to the solution
    
    while la.norm(r) > tolerence:
        Ap = np.dot(A, p)
        r_old = np.dot(r.T, r)
        
        alpha = r_old / np.dot(p.T, Ap) # Step size update        
        x += alpha * p                  # Update in solution x        
        r += alpha * Ap                 # Update in residual r
        beta = np.dot(r.T, r) / r_old
        p = -r + beta * p               # Update in conjugate direction p
        
        iterations += 1                 # Update in iteration count
        
        print(f"Iteration {iterations}: ||r|| = {round(la.norm(r), 10)}, Value of the objective function = {round((0.5 * np.dot(np.dot(x.T, A), x) - np.dot(b.T, x))[0][0], 4)}")

    return (x, iterations)


# Define nxn matrix A as per given specifications
def A(n):
    A = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            A[i - 1, j - 1] = 1 / (i + j - 1)
    return A

# Define right hand side vector b of size n
def b(n):
    return np.ones((n,1))

# Define initial approximation x0 of size n
def x0(n):
    return np.zeros((n,1))

tolerence = 1e-7

# Final output for different values of n
for n in [5, 8, 12, 20]:
    print(f"\nn = {n}")
    x, iterations = conjugate_gradient(A(n), b(n), x0(n), tolerence)
    print(f"\nThe solution with given tolerance level is \n {x} \n \nThe number of iterations required is {iterations}.\n")
