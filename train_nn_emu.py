import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import yaml
import h5py


class Emulator(tf.keras.Model):

    def __init__(self, n_params, nks, pc_sigmas, pc_mean, v,
                 n_hidden=[100, 100, 100], n_components=10):
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
        self.v = v

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


def train_emu(Ptrain, Ftrain, validation_frac=0.2,
              n_hidden=[100, 100, 100], n_pcs=20,
              n_epochs=1000):

    iis = np.random.rand(len(Ptrain)) > validation_frac

    Pval = Ptrain[~iis, :]
    Fval = Ftrain[~iis, :]

    Ptrain = Ptrain[iis, :]
    Ftrain = Ftrain[iis, :]

    # Construct Principle Components
    mean = np.mean(Ftrain, axis=0)
    mean = np.array(mean, dtype='float32')
    sigmas = np.std(Ftrain, axis=0)
    sigmas = np.array(sigmas, dtype='float32')
    Ftrain = (Ftrain - mean) / sigmas
    Fval = (Fval - mean) / sigmas

    cov_matrix = np.cov(Ftrain.T)
    w, v = np.linalg.eigh(cov_matrix)
    # flip to rank in ascending eigenvalue
    w = np.flip(w)
    v = np.flip(v, axis=1)
    v = np.array(v, dtype='float32')
    pc_train = np.dot(Ftrain, v)

    pc_mean = np.mean(pc_train, axis=0)
    pc_sigmas = np.std(pc_train, axis=0)

    # Learning rate and batch size schedule:

    lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    nbatchs = np.array([16, 32, 64, 128, 256]) * 10

    # Now start the emulator and run it
    emulator = Emulator(n_params=Ptrain.shape[-1], nks=Ftrain.shape[-1],
                        pc_sigmas=pc_sigmas, pc_mean=pc_mean, v=v,
                        n_components=n_pcs, n_hidden=n_hidden)
    emulator.compile(optimizer='adam', loss='mse', metrics=['mse'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    for lr, nbatch in zip(lrs, nbatchs):
        print("Using learning rate, batch size:  %.2e, %d." % (lr, nbatch))

        emulator.optimizer.lr = lr
        emulator.fit(Ptrain, Ftrain, epochs=n_epochs, batch_size=nbatch,
                     validation_data=(Pval, Fval), callbacks=[es], verbose=2)

    return emulator


if __name__ == '__main__':

    info_txt = sys.argv[1]
    with open(info_txt, 'rb') as fp:
        emu_info = yaml.load(fp)

    training_data_filename = emu_info['training_filename']
    output_path = emu_info['output_path']

    emu_target = emu_info['target']
    emu_params = emu_info['param_dataset']
    training_data = h5py.File(training_data_filename, 'r')

    n_hidden = emu_info['n_hidden']
    n_pcs = emu_info['n_pcs']
    use_asinh = emu_info['use_asinh']
    scale_by_std = emu_info['scale_by_std']

    # data should already be cleaned
    Ptrain = training_data[emu_params][:]
    Ftrain = training_data[emu_target][:]

    if use_asinh:
        if scale_by_std:
            Fstd = np.std(Ftrain, axis=0)
            Ftrain = np.arcsinh(Ftrain/Fstd)
        else:
            Ftrain = np.arcsinh(Ftrain)

    emu = train_emu(Ptrain, Ftrain, validation_frac=0.2,
                    n_hidden=n_hidden, n_pcs=n_pcs)

    emu.save(output_path)
