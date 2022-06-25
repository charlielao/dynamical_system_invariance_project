# %%
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from scipy.misc import derivative
import random
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from termcolor import colored
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# %%
dt = 1
t = tf.linspace(0, 30, int(30/dt))
x1 = tf.math.sin(t)
x2 = tf.math.sin(t+np.pi/4)
v1 = tf.math.cos(t)
v2 = tf.math.cos(t+np.pi/4)
plt.plot(t, x1, "--")
plt.plot(t, v1, "--")

# %%
X1 = tf.concat([x1[:,None], x2[:,None], v1[:,None], v2[:,None]], axis=-1)
X1 += tf.random.normal((X1.shape), 0, 0.1, dtype=tf.float64)
Y1 = (X1[2:,:]-X1[:-2, :])/(2) # to estimate acceleration and velocity by discrete differenation
X1 = X1[1:-1, :]
X = tf.concat([X1], axis=0)
Y = tf.concat([Y1], axis=0)

plt.plot(X[:,2])
plt.plot(Y[:,0])
# %%
test_range = 3 
test_density = 10
test_x1s = tf.linspace(-test_range,test_range,test_density)
test_x2s = tf.linspace(-test_range,test_range,test_density)
test_v1s = tf.linspace(-test_range,test_range,test_density)
test_v2s = tf.linspace(-test_range,test_range,test_density)
test_xx1, test_xx2, test_vv1, test_vv2 = tf.meshgrid(test_x1s, test_x2s, test_v1s, test_v2s)
test_points = tf.stack([tf.reshape(test_xx1,[-1]),tf.reshape(test_xx2,[-1]),tf.reshape(test_vv1,[-1]), tf.reshape(test_vv2,[-1])], axis=1)
# %%
class Zero_mean(gpflow.mean_functions.Constant):
    def __init__(self, output_dim: int = 1) -> None:
        gpflow.mean_functions.Constant.__init__(self)
        self.output_dim = output_dim
        del self.c

    def __call__(self, X) -> tf.Tensor:
        return tf.zeros((X.shape[0]*self.output_dim, 1), dtype=X.dtype)
def degree_of_freedom(kernel):
    K = kernel(test_points)
    return tf.linalg.trace(tf.tensordot(K, tf.linalg.inv(K+1e-6*tf.eye(K.shape[0], dtype=tf.float64)), 1))

# %%
class MOI4(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(active_dims=[0,1,2,3])
        self.Ka1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Ka2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        n = X.shape[0]
        m = X2.shape[0]

        zeros_nm =tf.zeros((n,m), dtype=tf.float64)

        Ka1_X1X2  = self.Ka1(X, X2) 
        Ka2_X1X2  = self.Ka2(X, X2) 
        Kv1_X1X2  = self.Kv1(X, X2) 
        Kv2_X1X2  = self.Kv2(X, X2) 

        K_X1X2   = tf.concat([tf.concat([Ka1_X1X2,zeros_nm, zeros_nm, zeros_nm],1),tf.concat([zeros_nm,Ka2_X1X2, zeros_nm, zeros_nm],1),tf.concat([zeros_nm, zeros_nm, Kv1_X1X2, zeros_nm],1),tf.concat([zeros_nm, zeros_nm, zeros_nm, Kv2_X1X2],1)],0)
        
        return K_X1X2 

    def K_diag(self, X):
        n = X.shape[0]
        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        
        Ka1_X  = self.Ka1(X) 
        Ka2_X  = self.Ka2(X) 
        Kv1_X  = self.Kv1(X) 
        Kv2_X  = self.Kv2(X) 

        K_X   = tf.concat([tf.concat([Ka1_X,zeros_nn, zeros_nn, zeros_nn],1),tf.concat([zeros_nn,Ka2_X, zeros_nn, zeros_nn],1),tf.concat([zeros_nn, zeros_nn, Kv1_X,zeros_nn],1),tf.concat([zeros_nn, zeros_nn, zeros_nn,Kv2_X],1)],0)
        
        return tf.linalg.tensor_diag_part(K_X)


# %%
moi = MOI4()
moi.Ka1.variance = gpflow.Parameter(moi.Ka1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
moi.Ka2.variance = gpflow.Parameter(moi.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
moi.Kv1.variance = gpflow.Parameter(moi.Kv1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
moi.Kv2.variance = gpflow.Parameter(moi.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
moi.Ka1.lengthscales = gpflow.Parameter(moi.Ka1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
moi.Ka2.lengthscales = gpflow.Parameter(moi.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
moi.Kv1.lengthscales = gpflow.Parameter(moi.Kv1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
moi.Kv2.lengthscales = gpflow.Parameter(moi.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 

# %%
m_normal = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,2,None], Y[:,3,None],Y[:,0, None], Y[:,1,None]],1)),(Y.shape[0]*4,1))), kernel=moi, mean_function=Zero_mean(output_dim=4))
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_normal.training_loss, m_normal.trainable_variables, options=dict(maxiter=100))
#pred, var = m_normal.predict_f(test_points)
print(m_normal.log_marginal_likelihood().numpy())
print_summary(m_normal)
# %%
class SHM2D_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invariance_range, invar_density, jitter_size):
        super().__init__(active_dims=[0, 1, 2, 3])
        self.Ka1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Ka2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.jitter = jitter_size
        invariance_x1s = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_x2s = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_v1s = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_v2s = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_xx1, invariance_xx2, invariance_vv1, invariance_vv2 = tf.meshgrid(invariance_x1s, invariance_x2s, invariance_v1s, invariance_v2s)
        self.invar_grids = tf.stack([tf.reshape(invariance_xx1,[-1]), tf.reshape(invariance_xx2,[-1]), tf.reshape(invariance_vv1,[-1]), tf.reshape(invariance_vv2,[-1])], axis=1)
        self.x1_g_squared = tf.tensordot(self.invar_grids[:,0,None],self.invar_grids[None,:,0],1)
        self.x2_g_squared = tf.tensordot(self.invar_grids[:,1,None],self.invar_grids[None,:,1],1)
        self.x1_g_dot_squared = tf.tensordot(self.invar_grids[:,2,None],self.invar_grids[None,:,2],1)
        self.x2_g_dot_squared = tf.tensordot(self.invar_grids[:,3,None],self.invar_grids[None,:,3],1)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        n = X.shape[0]
        m = X2.shape[0]

        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        zeros_nm =tf.zeros((n,m), dtype=tf.float64)
        zeros_mm =tf.zeros((m,m), dtype=tf.float64)
        
        Ka1_X1X1  = self.Ka1(X) 
        Ka2_X1X1  = self.Ka2(X) 
        Kv1_X1X1  = self.Kv1(X) 
        Kv2_X1X1  = self.Kv2(X) 
        K_X1X1   = tf.concat([tf.concat([Ka1_X1X1,zeros_nn, zeros_nn, zeros_nn],1),tf.concat([zeros_nn, Ka2_X1X1, zeros_nn, zeros_nn],1),tf.concat([zeros_nn, zeros_nn, Kv1_X1X1, zeros_nn],1),tf.concat([zeros_nn, zeros_nn, zeros_nn, Kv2_X1X1],1)],0)

        Ka1_X1X2  = self.Ka1(X, X2) 
        Ka2_X1X2  = self.Ka2(X, X2) 
        Kv1_X1X2  = self.Kv1(X, X2) 
        Kv2_X1X2  = self.Kv2(X, X2) 
        K_X1X2   = tf.concat([tf.concat([Ka1_X1X2,zeros_nm, zeros_nm, zeros_nm],1),tf.concat([zeros_nm, Ka2_X1X2, zeros_nm, zeros_nm],1),tf.concat([zeros_nm, zeros_nm, Kv1_X1X2, zeros_nm],1),tf.concat([zeros_nm, zeros_nm, zeros_nm, Kv2_X1X2],1)],0)
        
        K_X2X1   = tf.transpose(K_X1X2)
        
        Ka1_X2X2  = self.Ka1(X2) 
        Ka2_X2X2  = self.Ka2(X2) 
        Kv1_X2X2  = self.Kv1(X2) 
        Kv2_X2X2  = self.Kv2(X2) 
        K_X2X2   = tf.concat([tf.concat([Ka1_X2X2,zeros_mm, zeros_mm, zeros_mm],1),tf.concat([zeros_mm, Ka2_X2X2, zeros_mm, zeros_mm],1),tf.concat([zeros_mm, zeros_mm, Kv1_X2X2, zeros_mm],1),tf.concat([zeros_mm, zeros_mm, zeros_mm, Kv2_X2X2],1)],0)
        
        Ka1_X1Xg  = self.Ka1(X, self.invar_grids) 
        Ka2_X1Xg  = self.Ka2(X, self.invar_grids) 
        Kv1_X1Xg  = self.Kv1(X, self.invar_grids) 
        Kv2_X1Xg  = self.Kv2(X, self.invar_grids) 
        K_X1Xg = tf.concat([Ka1_X1Xg, Ka2_X1Xg, Kv1_X1Xg, Kv2_X1Xg],0)

        Ka1_X2Xg =  self.Ka1(X2, self.invar_grids) 
        Ka2_X2Xg =  self.Ka2(X2, self.invar_grids) 
        Kv1_X2Xg =  self.Kv1(X2, self.invar_grids)
        Kv2_X2Xg =  self.Kv2(X2, self.invar_grids)
        K_X2Xg = tf.concat([Ka1_X2Xg, Ka2_X2Xg, Kv1_X2Xg, Kv2_X2Xg],0)

        Ka1_XgXg = self.Ka1(self.invar_grids) 
        Ka2_XgXg = self.Ka2(self.invar_grids) 
        Kv1_XgXg = self.Kv1(self.invar_grids) 
        Kv2_XgXg = self.Kv2(self.invar_grids) 
        
        x1_g_1 = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,0]
        x2_g_1 = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x1_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,2]
        x2_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,3]
        x_g_1_stacked = tf.concat([x1_g_dot_1, x2_g_dot_1, x1_g_1, x2_g_1],0)
        
        x1_g_2 = tf.ones([m, 1], dtype=tf.float64) * self.invar_grids[:,0]
        x2_g_2 = tf.ones([m, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x1_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * self.invar_grids[:,2]
        x2_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * self.invar_grids[:,3]
        x_g_2_stacked = tf.concat([x1_g_dot_2, x2_g_dot_2, x1_g_2, x2_g_2],0)

        
        A = tf.concat([tf.concat([K_X1X1, K_X1X2],1),tf.concat([K_X2X1, K_X2X2],1)],0) 
        B1 = tf.multiply(K_X1Xg, x_g_1_stacked)
        B2 = tf.multiply(K_X2Xg, x_g_2_stacked)
        B = tf.concat([B1, B2], 0)
        C = tf.transpose(B)
        D = tf.multiply(self.x1_g_dot_squared, Ka1_XgXg) + tf.multiply(self.x2_g_dot_squared, Ka2_XgXg) + tf.multiply(self.x1_g_squared, Kv1_XgXg) + tf.multiply(self.x2_g_squared, Kv2_XgXg) 
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        
        return (A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), C, 1))[:4*n, 4*n:]

    def K_diag(self, X):
        n = X.shape[0]
        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        
        Ka1_X  = self.Ka1(X) 
        Ka2_X  = self.Ka2(X) 
        Kv1_X  = self.Kv1(X) 
        Kv2_X  = self.Kv2(X) 
        K_X   = tf.concat([tf.concat([Ka1_X, zeros_nn, zeros_nn, zeros_nn],1),tf.concat([zeros_nn, Ka2_X, zeros_nn, zeros_nn],1),tf.concat([zeros_nn, zeros_nn, Kv1_X, zeros_nn],1),tf.concat([zeros_nn, zeros_nn, zeros_nn, Kv2_X],1)],0)
        
        Ka1_Xg  = self.Ka1(X, self.invar_grids) 
        Ka2_Xg  = self.Ka2(X, self.invar_grids) 
        Kv1_Xg  = self.Kv1(X, self.invar_grids) 
        Kv2_Xg  = self.Kv2(X, self.invar_grids) 
        K_Xg = tf.concat([Ka1_Xg, Ka2_Xg, Kv1_Xg, Kv2_Xg],0)

        Ka1_XgXg = self.Ka1(self.invar_grids) 
        Ka2_XgXg = self.Ka2(self.invar_grids) 
        Kv1_XgXg = self.Kv1(self.invar_grids) 
        Kv2_XgXg = self.Kv2(self.invar_grids) 
        
        x1_g = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,0]
        x2_g = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x1_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,2]
        x2_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,3]
        x_g_stacked = tf.concat([x1_g_dot, x2_g_dot, x1_g, x2_g],0)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(self.x1_g_dot_squared, Ka1_XgXg) + tf.multiply(self.x2_g_dot_squared, Ka2_XgXg) + tf.multiply(self.x1_g_squared, Kv1_XgXg) + tf.multiply(self.x2_g_squared, Kv2_XgXg) 
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

# %%
energy_kernel = SHM2D_Invariance(3, 5, 1e-5)
energy_kernel.Ka1.variance = gpflow.Parameter(energy_kernel.Ka1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
energy_kernel.Ka2.variance = gpflow.Parameter(energy_kernel.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
energy_kernel.Kv1.variance = gpflow.Parameter(energy_kernel.Kv1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
energy_kernel.Kv2.variance = gpflow.Parameter(energy_kernel.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
energy_kernel.Ka1.lengthscales = gpflow.Parameter(energy_kernel.Ka1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
energy_kernel.Ka2.lengthscales = gpflow.Parameter(energy_kernel.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
energy_kernel.Kv1.lengthscales = gpflow.Parameter(energy_kernel.Kv1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
energy_kernel.Kv2.lengthscales = gpflow.Parameter(energy_kernel.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
# %%
# posterior
m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,2,None], Y[:,3,None],Y[:,0, None], Y[:,1,None]],1)),(Y.shape[0]*4,1))), kernel=energy_kernel, mean_function=Zero_mean(output_dim=4))

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
#pred, var = m.predict_f(test_points)
print(m.log_marginal_likelihood().numpy())
print_summary(m)