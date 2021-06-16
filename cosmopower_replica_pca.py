import numpy as np
import matplotlib.pyplot as plt

#
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping

# The function we want to emulate:
kvec = np.arange(0,1,0.01)
print(kvec.shape)

def func(a,b):
    return a * (2 + np.sin(2*np.pi*b*kvec) )

# Define sample:
Avec = np.arange(1,2,0.01)
Bvec = np.arange(1,2,0.01)

Agrid, Bgrid = np.meshgrid(Avec,Bvec)
Atrain, Btrain = Agrid.flatten(), Bgrid.flatten()
Ptrain = np.vstack( (Atrain,Btrain) ).T

Ftrain = [ func(Atrain[ii],Btrain[ii]) for ii in range(len(Atrain)) ]
Ftrain = np.array(Ftrain)

# Split into validation set 30-70
iis = np.random.rand(len(Atrain)) > 0.3

Pval = Ptrain[~iis,:]
Fval = Ftrain[~iis,:]

Ptrain = Ptrain[iis,:]
Ftrain = Ftrain[iis,:]


# Construct Principle Components
mean = np.mean(Ftrain,axis=0); mean = np.array(mean, dtype='float32')
sigmas = np.std(Ftrain,axis=0); sigmas = np.array(sigmas,dtype='float32')

cov_matrix = np.cov( ((Ftrain - mean)/ sigmas).T )
w, v = np.linalg.eigh(cov_matrix)
# flip to rank in ascending eigenvalue
w = np.flip(w)
v = np.flip(v, axis=1)

v = np.array(v, dtype='float32')


# Learning rate and batch size schedule:

lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
nbatchs = [16, 32, 64, 128, 256]

# These are in the paper... we don't have that many samples
#lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
#nbatchs = [1000, 10000, 20000, 30000,40000, 50000]

# Define the emulator class (PCA version):
# This part is basically taken line by line, with a few modifications from Alsing et al 2020
# https://github.com/justinalsing/speculator/blob/master/speculator/speculator.py

class Emulator(tf.keras.Model):
    
    def __init__(self):
        super(Emulator, self).__init__()
        
        trainable = True
        
        self.n_parameters = 2
        self.n_hidden = [100,100,100]
        self.n_components = 10
        self.nks = 100
        
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_components]
        self.n_layers = len(self.architecture) - 1

        self.W = []
        self.b = []
        self.alphas = []
        self.betas = [] 
        
        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., np.sqrt(2./self.n_parameters)), name="W_" + str(i), trainable=trainable))
            self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i), trainable=trainable))
        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i), trainable=trainable))
            self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i), trainable=trainable))
            
    def activation(self, x, alpha, beta):
        return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta)) ), x)
    
    @tf.function
    def call(self, parameters):
        
        outputs = []
        #layers = [tf.divide(tf.subtract(parameters, self.parameters_shift), self.parameters_scale)]
        
        x = parameters
        
        for i in range(self.n_layers - 1):
            
            # linear network operation
            x = tf.add(tf.matmul(x, self.W[i]), self.b[i])
            
            # non-linear activation function
            x = self.activation(x, self.alphas[i], self.betas[i])

        # linear output layer
        x = tf.add(tf.matmul(x, self.W[-1]), self.b[-1])
        
        x = tf.add(tf.multiply(sigmas,tf.matmul(x,v[:,:self.n_components].T)),mean)

        return x

# Now start the emulator and run it

emulator = Emulator()
emulator.compile(optimizer='adam',loss='mse',metrics=['mse'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

for lr, nbatch in zip(lrs,nbatchs):
    print("Using learning rate, batch size:  %.2e, %d."%(lr,nbatch))
    
    emulator.optimizer.lr = lr
    emulator.fit(Ptrain,Ftrain,epochs=1000,batch_size=nbatch,validation_data=(Pval,Fval),callbacks=[es],verbose=2)
    

# Test it on some random parameters:

a, b = 1.153, 1.881

Xtest = np.array([a,b])[None,:]
Yreal = func(a,b)

Ylearn = emulator.predict(Xtest)[0,:]
abserr = np.max(np.abs(Ylearn-Yreal))
maxloc = np.argmax(np.abs(Ylearn-Yreal))
print("Max.abs.error ",abserr," at x=",kvec[maxloc])

plt.figure(figsize=(10,8))

plt.subplot(2,1,1)
plt.plot(kvec, Ylearn, label='nn')
plt.plot(kvec, Yreal, '.', label='real')

plt.ylabel('P')
plt.legend()

plt.subplot(2,1,2)
plt.plot(kvec, Ylearn/Yreal - 1)
plt.ylim(-0.01,0.01)
plt.ylabel(r'$\Delta P/P$')
plt.xlabel('k')
plt.savefig('test_pca.pdf')

