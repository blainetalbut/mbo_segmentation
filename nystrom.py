import random
import numpy as np
import math
import scipy
from scipy import linalg
import matplotlib.pyplot as plt

### Calculate an SVD approximation to a Gram matrix ###

# g is an N x N positive definite matrix (positive definiteness not enforced)
# l is the number of vertices to be sampled, by default sqrt(N)
def nystrom(g, l=None):
    # random.seed()

    n,k = g.shape

    try: assert n==k
    except: raise Exception("the matrix passed to nystrom is not square!")

    if not l:
        l = int(math.sqrt(n))
    if l > n:
        l=n

    # samples = random.sample(range(n), l)
    # samples.sort() # for convenience
    # for now let's just take the first l entries, since the input is random anyway
    samples = range(l)
    leftovers = [ i for i in range(n) if i not in samples]

    w_xx = np.array( [[g[i,j] for j in samples] for i in samples] )
    # print('w_xx = {}'.format(w_xx))
    if np.isnan( np.linalg.inv(w_xx) ).any():
        print('w_xx^-1 = {}'.format(np.linalg.inv(w_xx)))
    w_xy = np.array( [[g[i,j] for j in leftovers] for i in samples] )
    # print('w_xy = {}'.format(w_xy))
    w_yx = w_xy.transpose()
    w_yy = np.array( [[g[i,j] for j in leftovers] for i in leftovers] )

    onex = np.atleast_2d(np.ones(l)).T
    oney = np.atleast_2d(np.ones(n-l)).T

    dx = w_xx @ onex + w_xy @ oney
    dy = w_yx @ onex + w_yx @ np.linalg.inv(w_xx) @ w_xy @ oney
    if np.isnan(dx).any():
        print('dx contains nan')
    if np.isnan(dy).any():
        print('dy contains nan')
    sx = np.sqrt(dx)
    sy = np.sqrt(dy)
    # print('sx = {0}, sy = {1}'.format(sx,sy))


    if np.isnan(sx).any():
        print('sx contains nan')
    if np.isnan(sy).any():
        print('sy contains nan')

    # ww_xx = np.divide(w_xx, sx @ sx.transpose())
    # ww_xy = np.divide(w_xy, sx @ sy.transpose())
    # ww_yx = np.divide(w_yx, sy @ sx.T)
    # ww_yx = ww_xy.T

    # the normalization above is used in bertozzi et al. only to relate the weight matrix to the Laplacian
    # for straight up approximation it is unnecessary
    ww_xx = w_xx
    ww_xy = w_xy
    ww_yx = w_yx

    # print('ww_xx = {}'.format(ww_xx))
    # print('ww_xy = {}'.format(ww_xy))
    # print('ww_yx = {}'.format(ww_yx))
    # print('ww_xy . ww_yx = {}'.format(ww_xy @ ww_yx))
    # print('gam diagonalizes {}'.format(ww_xx + np.linalg.inv(scipy.linalg.sqrtm(ww_xx)) @ ww_xy @ ww_yx @ np.linalg.inv(scipy.linalg.sqrtm(ww_xx))))

    bx, d, bxt = np.linalg.svd(ww_xx)
    at, gam, a = np.linalg.svd(ww_xx + np.linalg.inv(scipy.linalg.sqrtm(ww_xx)) @ ww_xy @ ww_yx @ np.linalg.inv(scipy.linalg.sqrtm(ww_xx)))
    # print('d = {}.'.format(d))
    # print('gam = {}'.format(gam))

    vmat = np.concatenate((
        bx @ np.diag(np.sqrt(d)) @ bxt @ a @ np.linalg.inv(np.diag(np.sqrt(gam))),
        ww_xy.transpose() @ bx @ np.linalg.inv(np.diag(np.sqrt(d))) @ bxt @ a @ np.linalg.inv(np.diag(np.sqrt(gam))) ))
    vals = gam
    

    return vmat, vals

def test(size = 100):
    rm = np.random.rand(size,size)

    # we calculate the standard deviation
    avg = np.sum(rm, 1) / size
    avg = np.atleast_2d(np.array(avg)).T
    stdv = np.sum([ np.sum( np.square( rm[i,:] - avg ) ) for i in range(size) ]) / size **2
    # generate a random similarity matrix (gaussian kernel with metric distance of random features)
    m = np.array([[ math.exp(-(1/(2*stdv)) * np.linalg.norm(rm[i,:] - rm[j,:]) ** 2 ) for j in range(size) ] for i in range(size) ])

    # m = np.diag([1] * size)
    # print('m = {}'.format(m))
    # print(np.linalg.det(m))
    vecs, vals = nystrom(m)
    a, d, at = np.linalg.svd(m)
    # print('actual eigenvalues are {}'.format(d))
    # print('reported: {}'.format(vals))
    # for i in range(len(vecs)):
        # print(np.linalg.det(m - np.diag([vals[i]]*size)))
        # # errs = [ abs(x - vals[i]) for x in np.divide(m @ vecs[i], vecs[i]) ]
        # # print('For {0}th eigenvalue, error is {1}'.format(i, max(errs)))
    s = np.concatenate(vecs, 1)
    # print(vecs)
    # print(s.shape)
    approx = s @ np.diag(vals) @ s.T
    print('frobenius error = {}'.format(np.linalg.norm(m - approx)))
    print('m = {}'.format(m))
    print('approximation is {}'.format(approx))

    # xs = range(size)
    # ys = []
    # for x in xs:
        # lrm = m.copy()
        # lrm.resize((x,x))
        # lrm.resize((size,size))
        # ys.append(np.linalg.norm(m-lrm))
    # plt.plot(xs,ys)
    # plt.show()

    # return vecs, s, m, approx

