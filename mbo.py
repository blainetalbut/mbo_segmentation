# implementation of the two-step scheme to minimize the Ginzburg-Landau functional for a graph potential
# the potential is encoded as an array of length [# of vertices]
# the method uses auxiliary weights a and d which are arrays of length [# of eigenvalues]

import numpy as np
import math
import nystrom
import scipy
from scipy import sparse
from scipy.sparse import linalg

def threshold(parr):
    outarr = np.array(parr)
    for k in range(len(parr)):
        if parr[k] > 0:
            outarr[k] = 1
        # elif parr[k] == 0:
            # outarr[k] = 0
        else:
            outarr[k] = -1
    return outarr

class MBO_segmenter:

    def __init__(self, g, parr0 = None, num_eigenvecs = None, dt = .01, C = 100, stopcond = 1e-7):
        self.graph = g
        if num_eigenvecs:
            self.num_eigenvecs = num_eigenvecs
        else:
            self.num_eigenvecs = max(2,int(math.sqrt(g.shape[0])))
        self.dt = dt
        self.C = C
        self.stopcond = stopcond

        self._eigenvals = None
        self._eigenvecs = None

        if not(parr0 is None):
            self.parr0 = parr0
        else:
            self.parr0 = self.Eigenvecs()[:,-2]


    def laplacian_spectrum(self): # sparsify all this!!
        # return the self.numeggr.eigenvectors largest eigenvectors
        if (self._eigenvecs is None) or (self._eigenvals is None):
            size = self.graph.shape[0]
            degr = np.array( [1] * size) @ self.graph
            dd = scipy.sparse.diags(degr ** -0.5)
            laplacian = scipy.sparse.diags([1] * size) - dd @ self.graph @ dd

            # the nystrom extension, even when it operates at full rank, gives weird singular eigenvectors
            # the straight up np.svd method works fine... nystrom needs more debugging
            # self._eigenvecs, self._eigenvals = nystrom.nystrom(laplacian, self.num_eigenvecs)

            # we should be using the largest eigenvalues (which='LM')
            self._eigenvecs, self._eigenvals, _ = sparse.linalg.svds(laplacian, which='LM', k=self.num_eigenvecs)
        return self._eigenvecs, self._eigenvals

    def Eigenvals(self):
        return self.laplacian_spectrum()[1]
    def Eigenvecs(self):
        return self.laplacian_spectrum()[0]

    def heatflow_step(self, warr, darr):
        eigenvarr, lambarr = self.laplacian_spectrum()
        xlen = len(eigenvarr[:,1])
        outdarr = np.array(darr)
        yarr = np.zeros(xlen)

        outwarr = np.divide(warr - self.dt * darr, 1 + self.dt * lambarr)
        for x in range(xlen):
            yarr[x] = np.sum(outwarr * eigenvarr[x,:])
        for k in range(len(warr)):
            outdarr[k] = np.sum(self.C * (yarr - self.parr0) * eigenvarr[:, k])
        print(f'outwarr = {outwarr}, yarr = {yarr}')

        return outwarr, outdarr, yarr

    def mbo_mean_zero(self, step_count = 10):
        eigenvecs, lambdas = self.laplacian_spectrum()
        weights = np.ones(len(lambdas))
        for k in range(len(lambdas)):
            weights[k] = np.sum(self.parr0 * eigenvecs[:,k])
        penalties = np.zeros(len(lambdas))
        error = 2 * self.stopcond
        old_u = self.parr0 - np.average(self.parr0)

        print(old_u)

        while error >= self.stopcond:
            for s in range(step_count): # performs badly if step_count=1
                weights, penalties, y = self.heatflow_step(weights, penalties)
            y = y - np.average(y)
            u = threshold(y)
            for k in range(len(lambdas)):
                weights[k] = np.sum(u * eigenvecs[:,k])
                penalties[k] = np.sum(self.C*(y - self.parr0) * eigenvecs[:,k])
            error = np.sum( (u - old_u) ** 2) / np.sum(u ** 2)
            print(error)
            old_u = u

        return u

    def mbo(self, step_count = 10):
        #binary segmentation
        #one class is +1, the other is -1

        eigenvecs = self.Eigenvecs()
        num_eigenvecs = eigenvecs.shape[1]
        weights = np.ones(num_eigenvecs)
        for k in range(num_eigenvecs):
            weights[k] = np.sum(self.parr0 * eigenvecs[:,k])
        penalties = np.zeros(num_eigenvecs)
        error = 2 * self.stopcond
        old_u = self.parr0

        while error >= self.stopcond:
            for s in range(step_count): # performs badly if step_count=1
                weights, penalties, y = self.heatflow_step(weights, penalties)
            u = threshold(y)
            for k in range(num_eigenvecs):
                weights[k] = np.sum(u * eigenvecs[:,k])
                penalties[k] = np.sum(self.C*(y - self.parr0) * eigenvecs[:,k])
            error = np.sum( (u - old_u) ** 2) / np.sum(u ** 2)
            print(f'error = {error}')
            old_u = u

        return u

    def run(self):
        return self.mbo()
