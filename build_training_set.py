#!/usr/bin/env python3
# 
# Build a training set for the emulator.
# This code reads the ranges of parameters from a file and
# then varies only those for which the upper and lower limit
# of the parameter are different.
#
import numpy           as     np
from   low_discrepancy import QuasiRandomSequence
from   real_space_pk   import AnzuPowerSpectra
from   mpi4py          import MPI
import sys


# Make these global.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc= comm.Get_size()




class AnzuParameters():
    """Returns a vector of parameters to sample at.
       This should be coordinated with the theory module being used,
       in this case for Anzu the parameters we can vary are:
           cosmo: wb,wc,ns,sig8,hub
           bias:  b1,b2,bs,bn
           stoch: sn
       and pmin,pmax should have bounds for each.
       To hold a parameter fixed set pmin=pmax for that parameter. """
    def __init__(self,pmin,pmax):
        """Takes the range of parameters, only varying those where pmin!=pmax.
        The dimensions of pmin and pmax should equal the total number
        of parameters for the model."""
        self.pmin = pmin
        self.pmax = pmax
        ndim      = np.sum(np.abs(pmax-pmin)>0)
        if rank==0: print("ndim=",ndim)
        self.qrs  = QuasiRandomSequence(ndim)
    def sample(self,n):
        """Gets the n'th vector of parameter values."""
        pars = pmin.copy()
        vary = np.abs(pmax-pmin)>0
        pars[vary] = pmin[vary]+(pmax-pmin)[vary]*self.qrs.get_vector(n)
        return(pars)
    #






def generate_models(params,nstart=0,nend=10):
    """Does the work of generating the model predictions.  This also
       needs to know something about what is passed upon model class
       creation and what is a parameter to the model's __call__ method."""
    par = None
    mod = None
    for n in range(nstart,nend):
        if n%nproc==rank:
            pvec = params.sample(n)
            if par is None:
                par = np.zeros( (nend-nstart,pvec.size) )
            model= AnzuPowerSpectra(pvec[0],pvec[1:6])
            thy  = model(pvec[6:])[1] # Just select Pgg as an example.
            if mod is None:
                mod = np.zeros( (nend-nstart,thy.size) )
            par[n,:] = pvec.copy()
            mod[n,:] = thy.copy()
    # Just reduce everything to process 0 for convenience.
    ptot,mtot = np.zeros_like(par),np.zeros_like(mod)
    comm.Reduce(par,ptot,op=MPI.SUM,root=0)
    comm.Reduce(mod,mtot,op=MPI.SUM,root=0)
    # Now write the answers to disk.  We have the choice here to
    # include constant parameters or exclude them.  I am including
    # them, as they can be easily removed later by selecting on
    # parameters with np.std==0.
    if rank==0:
        np.savetxt("par.txt.gz",ptot)
        np.savetxt("mod.txt.gz",mtot)
    #






if __name__=="__main__":
    # Call as "build par_limits_file", where par_limits_file is
    # a file of upper and lower parameter limits as in
    # e.g. anzu_simple.txt
    if len(sys.argv)!=2:
        print("Usage: "+sys.argv[0]+" <par_limits>",flush=True)
        comm.Abort(1) # Ensure the job actually dies.
    # Ideally would read this only from root and then Bcast, but
    # that's a bit more complicated since sizes are unknown.
    pmin,pmax = np.loadtxt(sys.argv[1],unpack=True)
    #
    pars = AnzuParameters(pmin,pmax)
    generate_models(pars,nstart=0,nend=10)
    #
