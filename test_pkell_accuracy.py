import sys, os
os.environ['COBAYA_NOMPI'] = 'True'
sys.path.append('/global/project/projectdirs/desi/users/jderose/CobayaLSS/')
sys.path.append('/global/project/projectdirs/desi/users/jderose/CobayaLSS/lss_likelihood/')
sys.path.append('/global/project/projectdirs/desi/users/jderose/EmulateLSS/')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from cobaya.model import get_model
from cobaya.yaml import yaml_load
import matplotlib
import yaml
import h5py
import json

plt.rcParams['figure.figsize']        = 8., 6.
plt.rcParams['figure.dpi']            = 100
plt.rcParams['figure.subplot.left']   = 0.125
plt.rcParams['figure.subplot.right']  = 0.9
plt.rcParams['figure.subplot.bottom'] = 0.125
plt.rcParams['figure.subplot.top']    = 0.9
plt.rcParams['axes.labelsize']        = 18
plt.rcParams['axes.titlesize']        = 18
plt.rcParams['xtick.top']             = True
plt.rcParams['xtick.bottom']          = True
plt.rcParams['ytick.left']            = True
plt.rcParams['ytick.right']           = True
plt.rcParams['xtick.direction']       = 'in'
plt.rcParams['ytick.direction']       = 'in'
plt.rcParams['xtick.labelsize']       = 18
plt.rcParams['ytick.labelsize']       = 18
plt.rcParams['xtick.major.pad']       = 6.
plt.rcParams['xtick.minor.pad']       = 6.
plt.rcParams['ytick.major.pad']       = 6.
plt.rcParams['ytick.minor.pad']       = 6.
plt.rcParams['xtick.major.size']      = 6. # major tick size in points
plt.rcParams['xtick.minor.size']      = 3. # minor tick size in points
plt.rcParams['ytick.major.size']      = 6. # major tick size in points
plt.rcParams['ytick.minor.size']      = 3. # minor tick size in points
plt.rcParams['text.usetex']           = True
plt.rcParams['font.family']           = 'serif'
plt.rcParams['font.serif']            = 'Computer Modern Roman Bold'
plt.rcParams['font.size']             = 18

class Emulator(tf.keras.Model):

    def __init__(self, n_params, nks, pc_sigmas, pc_mean, v,
                 n_hidden=[100, 100, 100], n_components=10,
                 mean=None, sigmas=None, fstd=None,
                 param_mean=None, param_sigmas=None):
        super(Emulator, self).__init__()

        trainable = True

        self.n_parameters = n_params
        self.n_hidden = n_hidden
        self.n_components = n_components
        self.nks = nks

        self.architecture = [self.n_parameters] + \
            self.n_hidden + [self.n_components]
        self.n_layers = len(self.architecture) - 1

        self.W = []
        self.b = []
        self.alphas = []
        self.betas = []
        self.pc_sigmas = pc_sigmas
        self.pc_mean = pc_mean
        self.param_mean = param_mean
        self.param_sigmas = param_sigmas
        self.v = v
        self.mean = mean
        self.sigmas = sigmas
        self.fstd = fstd

        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., np.sqrt(
                2./self.n_parameters)), name="W_" + str(i), trainable=trainable))
            self.b.append(tf.Variable(
                tf.zeros([self.architecture[i+1]]), name="b_" + str(i), trainable=trainable))
        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal(
                [self.architecture[i+1]]), name="alphas_" + str(i), trainable=trainable))
            self.betas.append(tf.Variable(tf.random.normal(
                [self.architecture[i+1]]), name="betas_" + str(i), trainable=trainable))

    def activation(self, x, alpha, beta):
        return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta))), x)

    @tf.function
    def call(self, parameters):

        outputs = []
        x = parameters

        for i in range(self.n_layers - 1):

            # linear network operation
            x = tf.add(tf.matmul(x, self.W[i]), self.b[i])

            # non-linear activation function
            x = self.activation(x, self.alphas[i], self.betas[i])

        # linear output layer
        x = tf.add(tf.multiply(tf.add(tf.matmul(
            x, self.W[-1]), self.b[-1]), self.pc_sigmas[:self.n_components]), self.pc_mean[:self.n_components])
        x = tf.matmul(x, self.v[:, :self.n_components].T)

        return x
    
    def save(self, filebase):
        W = [self.W.weights[i].numpy().tolist() for i in range(len(self.W.weights))]
        b = [self.b.weights[i].numpy().tolist() for i in range(len(self.b.weights))]
        alpha = [self.alphas.weights[i].numpy().tolist() for i in range(len(self.alphas.weights))]
        beta = [self.betas.weights[i].numpy().tolist() for i in range(len(self.betas.weights))]
        pc_sigmas = self.pc_sigmas[:self.n_components].tolist()
        pc_mean = self.pc_mean[:self.n_components].tolist()
        v = self.v[:,:self.n_components].tolist()

        with open('{}_W.json'.format(filebase), 'w') as fp:
            json.dump(W, fp)

        with open('{}_b.json'.format(filebase), 'w') as fp:
            json.dump(b, fp)

        with open('{}_alphas.json'.format(filebase), 'w') as fp:
            json.dump(alpha, fp)

        with open('{}_betas.json'.format(filebase), 'w') as fp:
            json.dump(beta, fp)

        with open('{}_pc_mean.json'.format(filebase), 'w') as fp:
            json.dump(pc_mean, fp)

        with open('{}_pc_sigmas.json'.format(filebase), 'w') as fp:
            json.dump(pc_sigmas, fp)

        with open('{}_v.json'.format(filebase), 'w') as fp:
            json.dump(v, fp)

        if hasattr(self, 'sigmas'):
            sigmas = self.sigmas.tolist()
            with open('{}_sigmas.json', 'w') as fp:
                json.dump(sigmas, fp)

        if hasattr(self, 'mean'):
            mean = self.mean.tolist()
            with open('{}_mean.json', 'w') as fp:
                json.dump(mean, fp)

        if hasattr(self, 'fstd'):
            fstd = self.fstd.tolist()
            with open('{}_fstd.json', 'w') as fp:
                json.dump(fstd, fp)
                
    def load(self, filebase):

        with open('{}_W.json'.format(filebase), 'r') as fp:
            self.W = json.load(fp)
            for i, wi in enumerate(self.W):
                self.W[i] = np.array(wi).astype(np.float32)

        with open('{}_b.json'.format(filebase), 'r') as fp:
            self.b = json.load(fp)
            for bi in self.b:
                bi = np.array(bi).astype(np.float32)

        with open('{}_alphas.json'.format(filebase), 'r') as fp:
            self.alphas = json.load(fp)
            for ai in self.alphas:
                ai = np.array(ai).astype(np.float32)

        with open('{}_betas.json'.format(filebase), 'r') as fp:
            self.betas = json.load(fp)
            for bi in self.betas:
                bi = np.array(bi).astype(np.float32)

        with open('{}_pc_mean.json'.format(filebase), 'r') as fp:
            self.pc_mean = np.array(json.load(fp)).astype(np.float32)

        with open('{}_pc_sigmas.json'.format(filebase), 'r') as fp:
            self.pc_sigmas = np.array(json.load(fp)).astype(np.float32)

        with open('{}_v.json'.format(filebase), 'r') as fp:
            self.v = np.array(json.load(fp)).astype(np.float32)

        with open('{}_sigmas.json'.format(filebase), 'r') as fp:
             self.sigmas = np.array(json.load(fp)).astype(np.float32)

        with open('{}_mean.json'.format(filebase), 'r') as fp:
             self.mean = np.array(json.load(fp)).astype(np.float32)

        with open('{}_fstd.json'.format(filebase), 'r') as fp:
             self.fstd = np.array(json.load(fp)).astype(np.float32)

        with open('{}_param_sigmas.json'.format(filebase), 'r') as fp:
             self.param_sigmas = np.array(json.load(fp)).astype(np.float32)

        with open('{}_param_mean.json'.format(filebase), 'r') as fp:
             self.param_mean = np.array(json.load(fp)).astype(np.float32)
                



def measure_accuracy(training_filename, target, 
                  target_params, modelpath,
                  n_pcs, val_factor=10):
    
    
    
    data = h5py.File(training_filename, 'r') 
    x = data[target_params][:][::val_factor]
    y = data[target][:][::val_factor]
    data.close()

    emu = Emulator(x.shape[-1], y.shape[-1],
                   None, None, None,
                   n_components=n_pcs)
    emu.load(modelpath)
    
    x_rescaled = (x - emu.param_mean) / emu.param_sigmas
    
    y_pred = np.sinh(emu(x_rescaled).numpy() * emu.sigmas + emu.mean) * emu.fstd
    err = np.abs((y - y_pred) / y)
    
    return err, y, y_pred


if __name__ == '__main__':
    info_txt = '/global/project/projectdirs/desi/users/jderose/EmulateLSS/configs/unit_redmagic_wl_x_rsd_allpars.yaml'
    with open(info_txt, 'rb') as fp:
        emu_info = yaml.load(fp)

    model = get_model(emu_info)

    emu_info = emu_info['emulate']
    training_data_filename = emu_info['output_filename']
    training_data = h5py.File(training_data_filename, 'r')
    
    targets = ['p0', 'p2', 'p4']
    modelpaths = ['/global/cscratch1/sd/jderose/cobaya_heft/chains/unit_redmagic_wl_x_rsd_allpars_broad_bias_100xfast_1e6pts_training_data_p0_emu',
                  '/global/cscratch1/sd/jderose/cobaya_heft/chains/unit_redmagic_wl_x_rsd_allpars_broad_bias_100xfast_1e6pts_training_data_p2_emu',
                  '/global/cscratch1/sd/jderose/cobaya_heft/chains/unit_redmagic_wl_x_rsd_allpars_broad_bias_100xfast_1e6pts_training_data_p4_emu']

    p0err, p0, p0_pred = measure_accuracy(training_data_filename, targets[0], 'params_pkell', modelpaths[0], 64)
    p2err, p2, p2_pred = measure_accuracy(training_data_filename, targets[1], 'params_pkell', modelpaths[1], 64)
    p4err, p4, p4_pred = measure_accuracy(training_data_filename, targets[2], 'params_pkell', modelpaths[2], 64)

    k = model.theory['theory_lss.RSD.RSDCalculator'].k

    f, ax = plt.subplots(3, 1, sharex=True, sharey=True)
    ax[0].loglog(k, np.percentile(p0err, [50, 68, 95], axis=0).T)
    ax[1].loglog(k, np.percentile(p2err, [50, 68, 95], axis=0).T)
    ax[2].loglog(k, np.percentile(p4err, [50, 68, 95], axis=0).T)


    ax[2].set_xlabel(r'$k \, [h\, Mpc^{-1}]$')
    ax[0].set_ylabel(r'$P_0(k)$ error')
    ax[1].set_ylabel(r'$P_2(k)$ error')
    ax[2].set_ylabel(r'$P_4(k)$ error')
    ax[0].set_ylim([1e-4, 5e-2])
    ax[0].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    ax[1].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    ax[2].get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    ax[0].legend(['50th percentile', '68th percentile', '95th percentile'], fontsize=12)
    plt.subplots_adjust(wspace=0.1, hspace=0.0)

    
    plt.savefig('pkell_accuracy.pdf', dpi=100, bbox_inches='tight')
    
