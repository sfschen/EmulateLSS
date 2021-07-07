import sys, os
os.environ['COBAYA_NOMPI'] = 'True'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
sys.path.append('/global/project/projectdirs/desi/users/jderose/CobayaLSS/')
sys.path.append('/global/project/projectdirs/desi/users/jderose/CobayaLSS/lss_likelihood/')
sys.path.append('/global/project/projectdirs/desi/users/jderose/EmulateLSS/')
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from cobaya.model import get_model
from cobaya.yaml import yaml_load
import yaml
import h5py

def load_training_data(key, dataset, downsample=None):
    
    keys = list(dataset.keys())
    keys = [k for k in keys if key in k]
    if downsample is not None:
        keys = keys[::downsample]
    nproc = len(keys)
    
    for i, k in enumerate(keys):
        rank = k.split('_')[-1]
        d = dataset[k][:]
        p = dataset['params_{}'.format(rank)][:]
        if i==0:
            size_i = d.shape
            size = [size_i[0]*nproc]
            psize = [size_i[0]*nproc, p.shape[1]]
            [size.append(size_i[i]) for i in range(1, len(size_i))]
            
            Y = np.zeros(size)
            X = np.zeros(psize)

        X[i * size_i[0]:(i + 1) * size_i[0]] = p
        Y[i * size_i[0]:(i + 1) * size_i[0]] = d
        
    return X, Y


if __name__ == '__main__':

    info_txt = sys.argv[1]
    with open(info_txt, 'rb') as fp:
        info = yaml.load(fp)
    
    info['packages_path'] = '/global/project/projectdirs/desi/users/jderose/CobayaLSS/'
    info['debug'] = False

    model = get_model(info)
    emu_info = info['emulate']
    training_data_filename = emu_info['output_filename']
    training_data = h5py.File(training_data_filename, 'r+')
    z = model.theory['theory_lss.HEFT.HEFTCalculator'].z

    Ptrain_all, Ftrain_all = load_training_data('lss_likelihood.wl_x_rsd.HarmonicSpaceWLxRSD.rs_power_spectra', training_data, downsample=2)

    idx = np.any(Ptrain_all>0, axis=1) #& np.all(Ftrain_all[:,0,:,:,1]>0, axis=(1,2))
    Ptrain = Ptrain_all[idx]
    Ftrain = Ftrain_all[idx]

    Ptrain_z = np.zeros((len(Ptrain)*len(z),Ptrain.shape[1]+1))
    for i in range(len(z)):
        Ptrain_z[i::len(z), :-1] = Ptrain
        Ptrain_z[i::len(z), -1] = z[i]
    Ptrain = Ptrain_z
    Ftrain = np.einsum('ijklm->ijkml', Ftrain).reshape(-1,300)
    cidx = [0,1,2,3,4,5,-1]
    Ptrain_cosmo = Ptrain[:,cidx]    
    try:
        training_data.create_dataset('params_pgg', Ptrain.shape)
        training_data.create_dataset('params_pmm', Ptrain_cosmo.shape)        
    except:
        del training_data['params_pgg']
        del training_data['params_pmm']
        del training_data['pmm']
        del training_data['pgm']
        del training_data['pgg']
        training_data.create_dataset('params_pgg', Ptrain.shape)
        training_data.create_dataset('params_pmm', Ptrain_cosmo.shape)

    training_data['params_pgg'][:] = Ptrain
    training_data['params_pmm'][:] = Ptrain_cosmo

    training_data.create_dataset('pgg', Ftrain[:,200:300].shape)
    training_data.create_dataset('pgm', Ftrain[:,200:300].shape)
    training_data.create_dataset('pmm', Ftrain[:,200:300].shape)

    training_data['pgg'][:] = Ftrain[:,200:300]
    training_data['pgm'][:] = Ftrain[:,100:200]
    training_data['pmm'][:] = Ftrain[:,:100]
    training_data.close()    
