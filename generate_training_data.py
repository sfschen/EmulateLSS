from cobaya.model import get_model
from cobaya.yaml import yaml_load
from low_discrepancy import QuasiRandomSequence
from mpi4py import MPI
import numpy as np
import yaml


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
        self.qrs = QuasiRandomSequence(ndim)

    def sample(self, n):
        """Gets the n'th vector of parameter values."""
        pars = self.pmin.copy()
        vary = np.abs(self.pmax-self.pmin) > 0
        pars[vary] = self.pmin[vary] + \
            (self.pmax-self.pmin)[vary]*self.qrs.get_vector(n)
        return pars


def generate_models(params, param_names, model, emu_info, nstart=0, nend=10):

    npars = nend - nstart
    out = {'params': np.zeros((npars, params.ndim))}

    for n in range(nstart, nend):
        if n % nproc == rank:
            pvec = params.sample(n)
            pars = dict(zip(param_names, pvec))
            out['params'][n] = pvec
            model.logposterior(pars)

            if 'provider' in emu_info:

                for thy in emu_info['provider']:
                    pred = model.provider.get_result(thy)

                    if thy not in out:
                        tsize = [d for d in pred.shape]
                        tsize.prepend(npars)
                        out[thy] = np.zeros(tsize)
                        out[thy][n] = pred

            if 'likelihood' in emu_info:

                for like in emu_info['likelihood']:
                    attr = emu_info['likelihood'][like]
                    pred = getattr(model.likelihood[like], attr)
                    name = '{}.{}'.format(like, attr)
                    if name not in out:
                        tsize = [d for d in pred.shape]
                        tsize.prepend(npars)
                        out[name] = np.zeros(tsize)
                        out[name][n] = pred

    for k in out:
        tot = np.zeros_like(out[k])
        comm.Reduce(out[k], tot, op=MPI.SUM, root=0)
        out[k] = tot

    if rank == 0:
        with h5py.File(emu_info['output_filename'], 'wb') as fp:

            for k in out:
                tot = out[k]
                shape = tot.shape
                fp.create_dataset(k, shape)
                fp[k] = tot


if __name__ == '__main__':

    info_txt = sys.argv[1]
    with open(info_txt, 'rb') as fp:
        info = yaml.load(fp)

    info['debug'] = False
    model = get_model(info)

    bounds = model.prior.bounds()
    param_names = model.parameters
    pars = AnzuParameters(bounds[:, 0], bounds[:, 1])
    emu_info = info['emulator']

    generate_models(pars, param_names, model, emu_info)
