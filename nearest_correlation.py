import numpy as np
from numpy import diag, inf
from numpy import copy, dot
from numpy.linalg import norm


class ExceededMaxIterationsError(Exception):
    def __init__(self, msg, matrix=[], iteration=[], ds=[]):
        self.msg = msg
        self.matrix = matrix
        self.iteration = iteration
        self.ds = ds

    def __str__(self):
        return repr(self.msg)


def nearcorr(A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
             weights=None, verbose=False,
             except_on_too_many_iterations=True):
    """
    X = nearcorr(A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
        weights=None, print=0)

    Finds the nearest correlation matrix to the symmetric matrix A.

    ARGUMENTS
    ~~~~~~~~~
    A is a symmetric numpy array or a ExceededMaxIterationsError object

    tol is a convergence tolerance, which defaults to 16*EPS.
    If using flag == 1, tol must be a size 2 tuple, with first component
    the convergence tolerance and second component a tolerance
    for defining "sufficiently positive" eigenvalues.

    flag = 0: solve using full eigendecomposition (EIG).
    flag = 1: treat as "highly non-positive definite A" and solve
    using partial eigendecomposition (EIGS). CURRENTLY NOT IMPLEMENTED

    max_iterations is the maximum number of iterations (default 100,
    but may need to be increased).

    n_pos_eig (optional) is the known number of positive eigenvalues
    of A. CURRENTLY NOT IMPLEMENTED

    weights is an optional vector defining a diagonal weight matrix diag(W).

    verbose = True for display of intermediate output.
    CURRENTLY NOT IMPLEMENTED

    except_on_too_many_iterations = True to raise an exeption when
    number of iterations exceeds max_iterations
    except_on_too_many_iterations = False to silently return the best result
    found after max_iterations number of iterations

    ABOUT
    ~~~~~~
    This is a Python port by Michael Croucher, November 2014
    Thanks to Vedran Sego for many useful comments and suggestions.

    Original MATLAB code by N. J. Higham, 13/6/01, updated 30/1/13.
    Reference:  N. J. Higham, Computing the nearest correlation
    matrix---A problem from finance. IMA J. Numer. Anal.,
    22(3):329-343, 2002.
    """

    # If input is an ExceededMaxIterationsError object this
    # is a restart computation
    if (isinstance(A, ExceededMaxIterationsError)):
        ds = copy(A.ds)
        A = copy(A.matrix)
    else:
        ds = np.zeros(np.shape(A))

    eps = np.spacing(1)
    if not np.all((np.transpose(A) == A)):
        raise ValueError('Input Matrix is not symmetric')
    if not tol:
        tol = eps * np.shape(A)[0] * np.array([1, 1])
    if weights is None:
        weights = np.ones(np.shape(A)[0])
    X = copy(A)
    Y = copy(A)
    rel_diffY = inf
    rel_diffX = inf
    rel_diffXY = inf

    Whalf = np.sqrt(np.outer(weights, weights))

    iteration = 0
    while max(rel_diffX, rel_diffY, rel_diffXY) > tol[0]:
        iteration += 1
        if iteration > max_iterations:
            if except_on_too_many_iterations:
                if max_iterations == 1:
                    message = "No solution found in "\
                              + str(max_iterations) + " iteration"
                else:
                    message = "No solution found in "\
                              + str(max_iterations) + " iterations"
                raise ExceededMaxIterationsError(message, X, iteration, ds)
            else:
                # exceptOnTooManyIterations is false so just silently
                # return the result even though it has not converged
                return X

        Xold = copy(X)
        R = X - ds
        R_wtd = Whalf*R
        if flag == 0:
            X = proj_spd(R_wtd)
        elif flag == 1:
            raise NotImplementedError("Setting 'flag' to 1 is currently\
                                 not implemented.")
        X = X / Whalf
        ds = X - R
        Yold = copy(Y)
        Y = copy(X)
        np.fill_diagonal(Y, 1)
        normY = norm(Y, 'fro')
        rel_diffX = norm(X - Xold, 'fro') / norm(X, 'fro')
        rel_diffY = norm(Y - Yold, 'fro') / normY
        rel_diffXY = norm(Y - X, 'fro') / normY

        X = copy(Y)

    return X


def proj_spd(A):
    # NOTE: the input matrix is assumed to be symmetric
    d, v = np.linalg.eigh(A)
    A = (v * np.maximum(d, 0)).dot(v.T)
    A = (A + A.T) / 2
    return(A)
