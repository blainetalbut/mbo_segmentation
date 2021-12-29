# implementation of the two-step scheme to minimize the Ginzburg-Landau functional for a graph potential
# the potential is encoded as an array of length [# of vertices]
# the method uses auxiliary weights a and d which are arrays of length [# of eigenvalues]

import numpy as np
import math
import nystrom

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

    def __init__(self, g, parr0 = None, num_eigenvecs = None, dt = 0.01, C = 10, stopcond = 1e-10):
        self.graph = g
        if num_eigenvecs:
            self.num_eigenvecs = num_eigenvecs
        else:
            self.num_eigenvecs = max(2,int(math.sqrt(g.shape[0])))
        self.dt = dt
        self.C = C
        self.stopcond = stopcond

        self.eigenvecs, self.eigenvals = nystrom.nystrom(self.graph, self.num_eigenvecs)

        if not(parr0 is None):
            self.parr0 = parr0
        else:
            self.parr0 = threshold(self.eigenvecs[:,1])

    def laplacian_spectrum(self):
        if (self.eigenvecs is None) or (self.eigenvals is None):
            self.eigenvecs, self.eigenvals = nystrom.nystrom(self.graph, self.num_eigenvecs)
        return self.eigenvecs, self.eigenvals

    def heatflow_step(self, warr, darr):
        eigenvarr, lambarr = self.laplacian_spectrum()
        outdarr = np.array(darr)
        parr = np.array(self.parr0)

        outwarr = np.divide(warr - self.dt * darr, 1 + self.dt * lambarr)
        for x in range(len(self.parr0)):
            parr[x] = np.sum(outwarr * eigenvarr[x,:])
        for k in range(len(warr)):
            outdarr[k] = np.sum(self.C * (parr - self.parr0) * eigenvarr[:, k])
        print(parr)

        parr = threshold(parr)
        for k in range(len(warr)):
            outwarr[k] = np.sum(parr * eigenvarr[:, k])

        return outwarr, outdarr, parr

    def run(self):
        eigenvarr, lambarr = self.laplacian_spectrum()
        warr = np.zeros(len(lambarr))

        for k in range(len(lambarr)):
            warr[k] = np.sum(self.parr0 * eigenvarr[:, k])
        print(warr)
        darr = np.zeros(len(lambarr))

        error = 2 * self.stopcond
        old_parr = self.parr0
        while error  >= self.stopcond :
            warr, darr, parr = self.heatflow_step(warr, darr)
            print(error)
            error = np.sum( (parr - old_parr) ** 2) / np.sum(parr ** 2)
            old_parr = parr
            
        return parr
