#!/usr/bin/env python3
#
# Classes that call various (real space) power spectrum models.
#
import numpy as np
import sys

# Model the (real-space) power spectrum using LPT.
from velocileptors.LPT.cleft_fftw import CLEFT
# Model the (real-space) power spectrum using Anzu.
from anzu.emu_funcs import LPTEmulator



class LPTPowerSpectra():
    """Computes the (real-space) power spectrum, P(k,z) [Mpc/h units]."""
    def combine_bias_terms_pk_crossmatter(self,b1,b2,bs,b3,alpha):
        """A CLEFT helper function to return P_{gm}."""
        kv  = self.cleft.pktable[:,0]
        ret = self.cleft.pktable[:,1]+0.5*b1*self.cleft.pktable[:,2]+\
              0.5*b2*self.cleft.pktable[:,4]+0.5*bs*self.cleft.pktable[:,7]+\
              0.5*b3*self.cleft.pktable[:,11]+\
              alpha*kv**2*self.cleft.pktable[:,13]
        return(kv,ret)
        #
    def calculate_pk(self,b1,b2,bs,b3,alpha_a,alpha_x,sn):
        """Returns k,Pgg,Pgm,Pmm using pre-computed CLEFT kernels."""
        kk,pgg = self.cleft.combine_bias_terms_pk(b1,b2,bs,b3,alpha_a,sn)
        kk,pgm = self.combine_bias_terms_pk_crossmatter(b1,b2,bs,b3,alpha_x)
        return((kk,pgg,pgm,self.cleft.pktable[:,1]))
        #
    def __init__(self,klin,plin):
        """klin,plin: Arrays containing Plinear [Mpc/h units]."""
        # Copy the arguments.
        self.klin = klin
        self.plin = plin
        # Set up the CLEFT class -- this can take a little while.
        self.cleft= CLEFT(self.klin,self.plin)
        self.cleft.make_ptable(nk=250)
        #
    def __call__(self,pk_pars):
        """Computes the three P(k,z).  Returns k,Pgg,Pgm,Pmm [Mpc/h units]."""
        return(self.calculate_pk(*pk_pars))
        #











class AnzuPowerSpectra():
    """Computes the (real-space) power spectrum, P(k,z) [Mpc/h units]."""
    def combine_bias_terms_pk_gg(self,b1,b2,bs,bn,sn):
        """A helper function to return P_{gg}."""
        # Order returned by Anzu is:
        #    1-1, delta-1 , delta-delta, delta2-1, delta2-delta, delta2-delta2
        #   s2-1,  s2-delta, s2-delta2, s2-s2,
        #   nd-nd, nd-delta, nd-delta2, nd-s2
        kv  = self.kv
        ret = 1.000*self.pktable[ 0,:] + 2   *b1*self.pktable[ 1,:]  +\
              1.*b2*self.pktable[ 3,:] + 2   *bs*self.pktable[ 6,:]  +\
              b1*b1*self.pktable[ 2,:] +   b1*b2*self.pktable[ 4,:]  +\
            2*b1*bs*self.pktable[ 7,:] +   b2*b2*self.pktable[ 5,:]/4+\
              b2*bs*self.pktable[ 8,:] +   bs*bs*self.pktable[ 9,:]  +\
              2.*bn*self.pktable[10,:] + 2*b1*bn*self.pktable[11,:]  +\
              b2*bn*self.pktable[12,:] + 2*bs*bn*self.pktable[13,:]  +\
              sn
        return(kv,ret)
        #
    def combine_bias_terms_pk_gm(self,b1,b2,bs,bn):
        """A helper function to return P_{gm}."""
        kv  = self.kv
        ret = 1.*self.pktable[ 0,:] + b1*self.pktable[1,:] +\
          0.5*b2*self.pktable[ 3,:] + bs*self.pktable[6,:] +\
              bn*self.pktable[10,:]
        return(kv,ret)
        #
    def combine_bias_terms_pk_mm(self):
        """A helper function to return P_{mm}."""
        kv  = self.kv
        ret = self.pktable[0,:]
        return(kv,ret)
        #
    def calculate_pk(self,b1,b2,bs,bn,sn):
        """Returns k,Pgg,Pgm,Pmm using Anzu."""
        # Combine pre-computed terms to get the spectra.
        kk,pgg = self.combine_bias_terms_pk_gg(b1,b2,bs,bn,sn)
        kk,pgm = self.combine_bias_terms_pk_gm(b1,b2,bs,bn)
        kk,pmm = self.combine_bias_terms_pk_mm()
        return((kk,pgg,pgm,pmm))
        #
    def __init__(self,z_eff,pars=None):
        """Initialize the class.
           z_eff is the (effective) redshift at which to evaluate P(k).
           If no cosmological parameters are passed it uses a default
           set otherwise "pars" should hold wb,wc,ns,sig8,hub."""
        # Store z_eff as a scale factor.
        self.aeff= 1.0/(1.0+z_eff)
        # Set the k-values we want to return.
        self.kv  = np.logspace(-3.0,0.0,256)
        # Set up the Anzu class
        self.emu = LPTEmulator(kecleft=True)
        # Fill in the basis spectra using Anzu -- this can take a little while.
        # Upon completion "pktable" is an (Nspec,Nk) array, extended assuming
        # <X, nabla^2 delta> ~ -k^2 <X, 1>.
        if pars is None:
            wb,wc,ns,sig8,hub = 0.022,0.119-0.0006442,0.96824-0.005,0.771,0.677
        else:
            wb,wc,ns,sig8,hub = pars
        cospars  = np.atleast_2d([wb,wc,-1.,ns,sig8,100*hub,3.046,self.aeff])
        pkvec    = self.emu.predict(self.kv,cospars)[0,:,:]
        # Now add the nabla terms assuming <X, nabla^2 delta> ~ -k^2 <X, 1>.
        # We ignore <nabla^2 del,nabla^2 del> so we have only 14 basis specta.
        self.pktable      = np.zeros( (14,len(self.kv)) )
        self.pktable[:10] = pkvec.copy()
        self.pktable[10:] = -self.kv**2 * pkvec[ [0,1,3,6] ]
        #
    def __call__(self,pk_pars):
        """Computes the three P(k,z).  Returns k,Pgg,Pgm,Pmm [Mpc/h units]."""
        return(self.calculate_pk(*pk_pars))
        #



