# Python versions of nearest correlation matrix algorithms.

This module will eventually contain several algorithms for solving nearest correlation matrix problems.

The only algorithm currently implemented is Nick Higham's. The code in this module is a port of the MATLAB original at http://nickhigham.wordpress.com/2013/02/13/the-nearest-correlation-matrix/

Run the test suite as follows:

```
python nearest_correlation_unittests.py
```

An example computation that finds the nearest correlation matrix to the input matrix:

```python
In [1]: from nearest_correlation import nearcorr

In [2]: import numpy as np

In [3]: A = np.array([[2, -1, 0, 0], 
   ...:               [-1, 2, -1, 0],
   ...:               [0, -1, 2, -1], 
   ...:               [0, 0, -1, 2]])

In [4]: X = nearcorr(A)

In [5]: X
Out[5]: 
array([[ 1.        , -0.8084125 ,  0.1915875 ,  0.10677505],
       [-0.8084125 ,  1.        , -0.65623269,  0.1915875 ],
       [ 0.1915875 , -0.65623269,  1.        , -0.8084125 ],
       [ 0.10677505,  0.1915875 , -0.8084125 ,  1.        ]])

```

Here's an example using the `weights` parameter. `weights` is a vector defining a diagonal weight matrix diag(W):.
```python
In [1]: from nearest_correlation import nearcorr

In [2]: import numpy as np

In [3]: A = np.array([[1, 1, 0],
   ...:               [1, 1, 1],
   ...:               [0, 1, 1]])

In [4]: weights = np.array([1,2,3])

In [5]: X = nearcorr(A, weights = weights)

In [6]: X
Out[6]: 
array([[ 1.        ,  0.66774961,  0.16723692],
       [ 0.66774961,  1.        ,  0.84557496],
       [ 0.16723692,  0.84557496,  1.        ]])
```

By default, the maximum number of iterations allowed before the algorithm gives up is 100.  This can be changed using the `max_iterations` parameter. When the number of iterations exceeds `max_iterations` an exception is raised unless `except_on_too_many_iterations = False`
```python
In [7]: A = np.array([[1, 1, 0],
   ...:               [1, 1, 1],
   ...:               [0, 1, 1]])

In [8]: nearcorr(A,max_iterations=10)
---------------------------------------------------------------------------
ExceededMaxIterationsError                Traceback (most recent call last)
<ipython-input-8-a79bc46a3452> in <module>()
----> 1 nearcorr(A,max_iterations=10)

/Users/walkingrandomly/Dropbox/nearest_correlation/nearest_correlation.py in nearcorr(A, tol, flag, max_iterations, n_pos_eig, weights, verbose, except_on_too_many_iterations)
    106                     message = "No solution found in "\
    107                               + str(max_iterations) + " iterations"
--> 108                 raise ExceededMaxIterationsError(message, X, iteration, ds)
    109             else:
    110                 # exceptOnTooManyIterations is false so just silently

ExceededMaxIterationsError: 'No solution found in 10 iterations'

```
If except_on_too_many_iterations=False, the best matrix found so far is quiety returned.
```python
In [10]: nearcorr(A,max_iterations=10,except_on_too_many_iterations=False)
Out[10]: 
array([[ 1.        ,  0.76073699,  0.15727601],
       [ 0.76073699,  1.        ,  0.76073699],
       [ 0.15727601,  0.76073699,  1.        ]])
```
#Continuing failed computations
If a computation failed because the the number of iterations exceeded `max_iterations`, it is possible to continue by passing the exception obejct to `nearcorr`:
```
from nearest_correlation import nearcorr, ExceededMaxIterationsError
import numpy as np

A = np.array([[1, 1, 0],
              [1, 1, 1],
              [0, 1, 1]])

# Is one iteration enough?
try:
    X = nearcorr(A, max_iterations=1)
except ExceededMaxIterationsError as e:
    restart = e # capture the Exception object
    print("1 iteration wasn't enough")

# start from where we left off
X = nearcorr(restart)

print(X)
```
