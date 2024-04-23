"""
Program the Newton algorithm using the backtracking line search, algorithm. Use them to minimize the Rosenbrock function:
                                      f(x) = 100(x2 − x1)^2 + (1 − x1)^2              (1)
To start with set the initial step length α0 = 1 and print the step length used by the each of the methods at each iteration. 
First try the initial point x0 = (-1.2, 1)^T. Trace the track followed by each of the algorithms step by step to the final 
point on a contour plot of the function in (1).
"""

import numpy as np
import numpy.linalg as la
import scipy.optimize as sopt
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

# Objective function
def f(x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Gradient
def df(x):
        return np.array ([-2 * (200 * x[0] * x[1] - 200 * x[0]**3 - x[0] + 1), 200 * (x[1] - x[0]**2)])

# Hessian
def ddf(x):
    return np.array([[-400 * x[1] + 1200 * x[0]**2, -400 * x[0]], [-400 * x[0], 200]])

fig = plt.figure()
ax = plt.axes(projection = '3d')

xmesh, ymesh = np.mgrid[-2:2:50j,-2:2:50j]
fmesh = f(np.array([xmesh, ymesh]))
ax.plot_surface(xmesh, ymesh, fmesh)

plt.show()

tol = 1e-4
iteration = 20
y0 = np.array ([-1.2, 1])   # initial guess
i = 0
y = [np.array(y0)]          # array of iterates produced

while la.norm(df(y[i])) > tol:
    
# descent direction = Newton direction
        pk = -np.dot(np.linalg.inv(ddf(y[i])), df(y[i])) 
    
# Backtracking algorithm
        rho = 0.9           # contraction factor
        alpha = 1           # initial guess for step length
        c = 1e-4
        j = 0
        
        while f(y[i] + alpha * pk) > f(y[i]) + c * alpha * np.dot(df(y[i]), pk):
                alpha = alpha * rho
                j +=1

        y1 = y[i] + alpha * pk
        y.append(y1)
        i += 1
        
        print("\nNumber of iteration to satisfy the Wolfe Condition is:", j)
        print("The",i,"th iterate is:", y[i])
        print("Norm of gradient at", i,"th iterate is:", la.norm(df(y[i])))


print("\nNorm of gradient at the solution =", la.norm(df(y[i])))
print("The value of function at the solution =", f(y[i]))
print("The number of iterations taken =", i)

plt.axis("equal")
plt.contour(xmesh, ymesh, fmesh, 50)
it_array = np.array(y)
plt.plot(it_array.T[0], it_array.T[1], "x-")
plt.show()
