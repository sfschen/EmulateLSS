import numpy as np
from scipy.special import expit
import json


class Emulator(object):

    def __init__(self, filebase, kmin=1e-3, kmax=0.5):
        super(Emulator, self).__init__()

        self.load(filebase)

        self.n_parameters = self.W[0].shape[0]
        self.n_components = self.W[-1].shape[-1]
        self.n_layers = len(self.W)
        self.nk = self.sigmas.shape[0]
        self.k = np.logspace(np.log10(kmin), np.log10(kmax), self.nk)

    def load(self, filebase):
        
        with open('{}.json'.format(filebase), 'r') as fp:
            weights = json.load(fp)
            
            for k in weights:
                if k in ['W', 'b', 'alphas', 'betas']:
                    for i, wi in enumerate(weights[k]):
                        weights[k][i] = np.array(wi).astype(np.float32)
                else:
                    weights[k] = np.array(weights[k]).astype(np.float32)

                setattr(self,k, weights[k])

    def activation(self, x, alpha, beta):
        return (beta + (expit(alpha * x) * (1 - beta))) * x

    def __call__(self, parameters):

        outputs = []
        x = (parameters - self.param_mean) / self.param_sigmas

        for i in range(self.n_layers - 1):

            # linear network operation
            x = x @ self.W[i] + self.b[i]

            # non-linear activation function
            x = self.activation(x, self.alphas[i], self.betas[i])

        # linear output layer
        x = ((x @ self.W[-1]) + self.b[-1]) * \
            self.pc_sigmas[:self.n_components] + \
            self.pc_mean[:self.n_components]
        x = np.sinh((x @ self.v[:, :self.n_components].T)
                    * self.sigmas + self.mean) * self.fstd

        return self.k, x
