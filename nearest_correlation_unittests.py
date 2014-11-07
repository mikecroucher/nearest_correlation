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

        (X, iter) = nearcorr(A)

        expected_result = np.array([[ 1.        , -0.8084125 ,  0.1915875 ,  0.10677505],
                                    [-0.8084125 ,  1.        , -0.65623269,  0.1915875 ],
                                    [ 0.1915875 , -0.65623269,  1.        , -0.8084125 ],
                                    [ 0.10677505,  0.1915875 , -0.8084125 ,  1.        ]])

        self.assertTrue((np.abs((X - expected_result)) < 1e-8).all())

    # This example taken from [1]
    def test_HighamExample2002(self):

        A = np.array([[1,1,0],
                      [1,1,1],
                      [0,1,1]])

        (X, iter) = nearcorr(A)

        expected_result = np.array([[ 1.        ,  0.76068985,  0.15729811],
                                    [ 0.76068985,  1.        ,  0.76068985],
                                    [ 0.15729811,  0.76068985,  1.        ]])

        self.assertTrue((np.abs((X - expected_result)) < 1e-8).all())

    # This uses the same input matrix as test_HighamExample2002
    # but I made up the weights vector since I couldn't find an example. No idea if it makes sense or not
    # Higham's MATLAB original was used as an oracle
    def test_Weights(self):
        A = np.array([[1,1,0],
                      [1,1,1],
                      [0,1,1]])

        weights = np.array([1,2,3])

        (X, iter) = nearcorr(A, weights = weights)

        expected_result = np.array([[ 1.        , 0.66774961, 0.16723692],
                                    [ 0.66774961, 1.        , 0.84557496],
                                    [ 0.16723692, 0.84557496, 1.        ]])

        self.assertTrue((np.abs((X - expected_result)) < 1e-8).all())

class InterfaceTests(unittest.TestCase):

    def test_AssertSymmetric(self):

         # Create a matrix that isn't symmetric
         A = np.array([[1,1,0],
                      [1,1,1],
                      [1,1,1]])

         # Ensure an exception is raised
         self.assertRaises(nearest_correlation.NotSymmetric,nearcorr,A)


def main():
    unittest.main()

if __name__ == '__main__':
    main()