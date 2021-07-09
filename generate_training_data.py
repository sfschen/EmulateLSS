from cobaya.model import get_model
from cobaya.yaml import yaml_load
from low_discrepancy import QuasiRandomSequence
from mpi4py import MPI
import numpy as np
import h5py
import yaml
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


class AnzuParameters():
    """Returns a vector of parameters to sample at."""

    def __init__(self, pmin, pmax):
        """Takes the range of parameters, only varying those where pmin!=pmax.
           The dimensions of pmin and pmax should equal the total number
           of parameters for the model."""
        self.pmin = pmin
        self.pmax = pmax
        self.ndim = np.sum(np.abs(pmax-pmin) > 0)
        self.qrs = QuasiRandomSequence(self.ndim)

    def sample(self, n):
        """Gets the n'th vector of parameter values."""
        pars = self.pmin.copy()
        vary = np.abs(self.pmax-self.pmin) > 0
        pars[vary] = self.pmin[vary] + \
            (self.pmax-self.pmin)[vary]*self.qrs.get_vector(n)
        return pars


def generate_models(params, param_names, model, emu_info, nstart=0, nend=10,
                    params_fast=None, param_names_fast=None, nfast_per_slow=1):

    if params_fast is None:
        allslow = True
        if nfast_per_slow != 1:
            print('Setting nfast_per_slow=1')
            nfast_per_slow = 1
    else:
        allslow = False
        assert(nfast_per_slow > 1) #otherwise no point in fast
        
    npars = nend - nstart
    
    if param_names_fast is not None:
        param_names_all = param_names + param_names_fast
    else:
        param_names_all = params_names
        
    npars_slow = (npars + nfast_per_slow - 1) // nfast_per_slow
    npars_this = ((npars_slow + nproc - 1) // nproc) * nfast_per_slow
    
    out = {'params': np.zeros((npars_this, len(param_names_all)))}    
    count = 0
    for n in range(npars_slow):
        if n % nproc == rank:
            if rank==0:
                print(n)
                sys.stdout.flush()

            pslow = params.sample(n)
            for m in range(nfast_per_slow):
                if not allslow:
                    fvec = params_fast.sample(n * nfast_per_slow + m)
                    pvec = np.concatenate([pslow, fvec])
                else:
                    pvec = pslow

                pars = dict(zip(param_names_all, pvec))
                out['params'][count] = pvec
                model.logposterior(pars)

                if 'provider' in emu_info:
                    for thy in emu_info['provider']:
                        pred = model.provider.get_result(thy)
                        
                        if thy not in out:
                            tsize = [npars_this]
                            [tsize.append(d) for d in pred.shape]
                            out[thy] = np.zeros(tsize)
                        out[thy][count] = pred

                if 'likelihood' in emu_info:
                    for like in emu_info['likelihood']:
                        if hasattr(emu_info['likelihood'][like], '__iter__'):
                            for l in emu_info['likelihood'][like]:
                                pred = getattr(model.likelihood[like], l)
                                name = '{}.{}'.format(like, l)
                                if name not in out:
                                    tsize = [npars_this]
                                    [tsize.append(d) for d in pred.shape]
                                    out[name] = np.zeros(tsize)
                                out[name][count] = pred
                            
                        else:
                            attr = emu_info['likelihood'][like]
                            pred = getattr(model.likelihood[like], attr)
                            name = '{}.{}'.format(like, attr)
                            if name not in out:
                                tsize = [npars_this]
                                [tsize.append(d) for d in pred.shape]
                                out[name] = np.zeros(tsize)
                            out[name][count] = pred
                count += 1


    for k in out:        
        with h5py.File('{}.{}'.format(emu_info['output_filename'], rank), 'w') as fp:
            for k in out:
                shape = out[k].shape
                fp.create_dataset(k, shape)
                fp[k][:] = out[k]            

    comm.Barrier()
    if rank==0:
        with h5py.File(emu_info['output_filename'], 'w') as fp:
            for k in out:
                for n in range(nproc):
                    fp['{}_{}'.format(k, n)] = h5py.ExternalLink('{}.{}'.format(emu_info['output_filename'], n), k)

if __name__ == '__main__':

    info_txt = sys.argv[1]
    with open(info_txt, 'rb') as fp:
        info = yaml.load(fp)

    info['debug'] = False
    model = get_model(info)

    bounds = model.prior.bounds()
    param_names = model.prior.params

    emu_info = info['emulate']
    nstart = emu_info.pop('nstart', 0)
    nend = emu_info.pop('nend', 100)
    param_names_fast = emu_info.pop('param_names_fast', None)
    nfast = emu_info.pop('nfast_per_slow', 1)

    if param_names_fast is not None:
        param_names = [p for p in param_names if p not in param_names_fast]
        fast_idx = [model.prior.params.index(f) for f in param_names_fast]
        slow_idx = [model.prior.params.index(f) for f in param_names]

        bounds_fast = bounds[fast_idx]
        bounds_slow = bounds[slow_idx]

        params = AnzuParameters(bounds_slow[:, 0], bounds_slow[:, 1])
        params_fast = AnzuParameters(bounds_fast[:, 0], bounds_fast[:, 1])
    else:
        params_fast = None
    
    generate_models(params, param_names, model, emu_info, nstart=nstart, nend=nend,
                    params_fast=params_fast, param_names_fast=param_names_fast,
                    nfast_per_slow=nfast)


