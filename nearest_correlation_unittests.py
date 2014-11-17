import unittest
import numpy as np
import nearest_correlation
from nearest_correlation import nearcorr

# References
# [1] 'Computing the nearest correlation matrix - a problem from finance': Higham, IMA Journal of Numerical Analysis (2002) 22, 329.343


class ResultsTests(unittest.TestCase):

    # This test is taken from the example given in the
    # NAG Mark 24 documentation for g02aa
    # It originally appeared in [1]
    def test_NAGExample(self):
        A = np.array([[2, -1, 0, 0], 
                      [-1, 2, -1, 0],
                      [0, -1, 2, -1], 
                      [0, 0, -1, 2]])

        X = nearcorr(A)

        expected_result = np.array([[ 1.        , -0.8084125 ,  0.1915875 ,  0.10677505],
                                    [-0.8084125 ,  1.        , -0.65623269,  0.1915875 ],
                                    [ 0.1915875 , -0.65623269,  1.        , -0.8084125 ],
                                    [ 0.10677505,  0.1915875 , -0.8084125 ,  1.        ]])

        self.assertTrue((np.abs((X - expected_result)) < 1e-8).all())

    # This example taken from [1]
    def test_HighamExample2002(self):

        A = np.array([[1, 1, 0],
                      [1, 1, 1],
                      [0, 1, 1]])

        X = nearcorr(A)

        expected_result = np.array([[ 1.        ,  0.76068985,  0.15729811],
                                    [ 0.76068985,  1.        ,  0.76068985],
                                    [ 0.15729811,  0.76068985,  1.        ]])

        self.assertTrue((np.abs((X - expected_result)) < 1e-8).all())

    # This uses the same input matrix as test_HighamExample2002
    # but I made up the weights vector since I couldn't find an example. No idea if it makes sense or not
    # Higham's MATLAB original was used as an oracle
    def test_Weights(self):
        A = np.array([[1, 1, 0],
                      [1, 1, 1],
                      [0, 1, 1]])

        weights = np.array([1,2,3])

        X = nearcorr(A, weights = weights)

        expected_result = np.array([[ 1.        , 0.66774961, 0.16723692],
                                    [ 0.66774961, 1.        , 0.84557496],
                                    [ 0.16723692, 0.84557496, 1.        ]])

        self.assertTrue((np.abs((X - expected_result)) < 1e-8).all())

    # A single calculation that fails after 3 iterations should give the same result as three calculations 
    # that each perform 1 iteration, restarting where they left off
    def test_restart(self):

        A = np.array([[1, 1, 0],
                      [1, 1, 1],
                      [0, 1, 1]])

        # Do 3 iterations on A and gather the result
        try:
            Y = nearcorr(A, max_iterations=3)
        except nearest_correlation.ExceededMaxIterationsError as e:
          result3 = np.copy(e.matrix)

        # Do 1 iteration on A
        try:
            X = nearcorr(A, max_iterations=1)
        except nearest_correlation.ExceededMaxIterationsError as e:
            restart = e

        # restart from previous result and do another iteration
        try:
            X = nearcorr(restart, max_iterations=1)
        except nearest_correlation.ExceededMaxIterationsError as e:
            restart = e

        # restart from previous result and do another iteration
        try:
            X = nearcorr(restart, max_iterations=1)
        except nearest_correlation.ExceededMaxIterationsError as e:
            result1 = e.matrix

        self.assertTrue(np.all(result1 == result3))


class InterfaceTests(unittest.TestCase):

    # Ensure that an exception is raised when a non-symmetric matrix is passed
    def test_AssertSymmetric(self):

        A = np.array([[1,1,0],
                      [1,1,1],
                      [1,1,1]])

        self.assertRaises(ValueError,nearcorr,A)


    # Ensure that an exception is raised when calculation does not converge befer maxiterations is exceeded
    def test_ExceededMaxIterations(self):
        A = np.array([[1,1,0],
                      [1,1,1],
                      [0,1,1]])

        self.assertRaises(nearest_correlation.ExceededMaxIterationsError,nearcorr,A,max_iterations=10)


    # Ensure that an exception is not raised when calculation does not converge befer maxiterations is exceeded
    # and except_on_too_many_iterations = False
    def test_ExceededMaxIterationsFalse(self):
        A = np.array([[1,1,0],
                      [1,1,1],
                      [0,1,1]])

        X = nearcorr(A,max_iterations=10,except_on_too_many_iterations=False)

def main():
    unittest.main()

if __name__ == '__main__':
    main()