# Python versions of nearest correlation matrix algorithms.

This module will eventually contain several algorithms for solving nearest correlation matrix problems.

The only algorithm currently implemented is Nick Higham's. The code in this module is a port of the MATLAB original at http://nickhigham.wordpress.com/2013/02/13/the-nearest-correlation-matrix/

Run the (currently rather small) test suite as follows:

```
python nearest_correlation_unittests.py
```

An example computation that finds the nearest correlation matrix to the input in 36 iterations:

```python
In [1]: from nearest_correlation import nearcorr

In [2]: import numpy as np

In [3]: A = np.array([[2, -1, 0, 0], 
   ...:               [-1, 2, -1, 0],
   ...:               [0, -1, 2, -1], 
   ...:               [0, 0, -1, 2]])

In [4]: (X, iter) = nearcorr(A)

In [5]: X
Out[5]: 
array([[ 1.        , -0.8084125 ,  0.1915875 ,  0.10677505],
       [-0.8084125 ,  1.        , -0.65623269,  0.1915875 ],
       [ 0.1915875 , -0.65623269,  1.        , -0.8084125 ],
       [ 0.10677505,  0.1915875 , -0.8084125 ,  1.        ]])

In [6]: iter
Out[6]: 36

```