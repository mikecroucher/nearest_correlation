import numpy as np
from numpy import diag, inf
from numpy import copy, dot
from numpy.linalg import norm


class NotSymmetric(Exception):
    pass


class NotImplemented(Exception):
    pass


def nearcorr(A, tol=[], flag=0, maxits=100, n_pos_eig=0,
             weights=np.array([]), prnt=0):
    eps = np.spacing(1)
    if not np.all((np.transpose(A) == A)):
        raise NotSymmetric('Input Matrix is not symmetric')
    if not tol:
        tol = eps * np.shape(A)[0] * np.array([1, 1])
    if weights.size == 0:
        weights = np.ones((np.shape(A)[0], 1))
    X = copy(A)
    Y = copy(A)
    iter = 1
    rel_diffY = inf
    rel_diffX = inf
    rel_diffXY = inf
    dS = np.zeros(np.shape(A))

    Whalf = np.sqrt(np.outer(weights, weights))  # Whalf = sqrt(w*w');

    while max(rel_diffX, rel_diffY, rel_diffXY) > tol[0]:
        Xold = copy(X)
        R = X - dS
        R_wtd = Whalf*R
        if flag == 0:
            X = proj_spd(R_wtd)
        elif flag == 1:
            raise
        X = X / Whalf
        dS = X - R
        Yold = copy(Y)
        Y = copy(X)
        np.fill_diagonal(Y, 1)  # Y = proj_unitdiag(X)
        normY = norm(Y, 'fro')
        rel_diffX = norm(X - Xold, 'fro') / norm(X, 'fro')
        rel_diffY = norm(Y - Yold, 'fro') / normY
        rel_diffXY = norm(Y - X, 'fro') / normY

        iter = iter + 1
        if iter > maxits:
            print("Too many iterations")
            return X, iter
        X = copy(Y)

    return X, iter


def proj_spd(A):
    d, v = np.linalg.eigh(A)
    A = v.dot(diag(nonneg(d))).dot(v.conj().T)
    A = (A + A.conj().T) / 2
    return(A)


def nonneg(A):
    B = copy(A)
    B[B < 0] = 0
    return(B)

    # test=np.random.uniform(0,1,(100,100))
    # sym=(test+test.T)/2
    # (X1,iter)=nearcorr(sym)
