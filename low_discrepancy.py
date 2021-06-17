#!/usr/bin/env python3
#
# Some simple Python code to make a low discrepancy sequence
# in multiple dimensions based upon:
# http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
#

import numpy as np


class QuasiRandomSequence():
    """A simple class to generate quasi-random, or low-discrepancy, sequences
       in multiple dimensions.  Based upon
       http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
       This is currently written in pure Python, but if speed ever becomes
       an issue it could likely be heavily optimized using Numba.
    """
    def __init__(self,n_dim=1,seed=0.5):
        """Initializes the class and sets the vector alpha.  The sequences
           will consist of n_dim-dimensional vectors in [0,1)^n_dim."""
        # Get the generalized golden ratio for n_dim dimensions, aka
        # the "harmonius numbers" of Hans van de Laan.
        self.seed= seed
        self.phi = 2
        for i in range(16):
            self.phi = pow(1+self.phi,1/(n_dim+1.0))
        self.phi = 1.0/self.phi
        # and hence generate our base vector, alpha.
        self.alpha=np.array([(self.phi**(i+1))%1 for i in range(n_dim)])
        #
    def __call__(self,n):
        """Returns the first n vectors in the (Korobov) sequence, with x[i,:]
           being the i'th vector."""
        tmp = self.seed + 0*self.alpha
        ret = np.zeros( (n,self.alpha.size) )
        for i in range(n):
            tmp     += self.alpha
            ret[i,:] = tmp
        ret = ret%1
        return(ret)
        #
    def nth_vector(self,n):
        """Returns just the nth vector in the sequence, starting at n=1."""
        ret = (self.seed + n*self.alpha)%1
        return(ret)
        #
