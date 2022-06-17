import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from termcolor import colored
class MOI(gpflow.kernels.Kernel):
    def __init__(self, jitter_size):
        super().__init__(active_dims=[0,1])
        self.jitter = gpflow.kernels.White(jitter_size)
        self.RBFa = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.RBFv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Ka =  self.RBFa + self.jitter
        self.Kv =  self.RBFv + self.jitter
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        n = X.shape[0]
        m = X2.shape[0]

        zeros_nm =tf.zeros((n,m), dtype=tf.float64)

        Ka_X1X2  = self.Ka(X, X2) 
        Kv_X1X2  = self.Kv(X, X2) 
        K_X1X2   = tf.concat([tf.concat([Ka_X1X2,zeros_nm],1),tf.concat([zeros_nm,Kv_X1X2],1)],0)
        
        return K_X1X2 

    def K_diag(self, X):
        n = X.shape[0]
        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        
        Ka_X  = self.Ka(X) 
        Kv_X  = self.Kv(X) 
        K_X   = tf.concat([tf.concat([Ka_X,zeros_nn],1),tf.concat([zeros_nn,Kv_X],1)],0)
        
        return tf.linalg.tensor_diag_part(K_X)

class SHM_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invariance_range, invar_density, jitter_size):
        super().__init__(active_dims=[0, 1])
        self.jitter = gpflow.kernels.White(jitter_size)
        self.RBFa = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.RBFv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Ka =  self.RBFa + self.jitter
        self.Kv =  self.RBFv + self.jitter
        invariance_xs = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_vs = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_xx, invariance_vv = tf.meshgrid(invariance_xs, invariance_vs)
        self.invar_grids = tf.stack([tf.reshape(invariance_xx,[-1]), tf.reshape(invariance_vv,[-1])], axis=1)
        self.x_g_squared = tf.tensordot(self.invar_grids[:,0,None],self.invar_grids[None,:,0],1)
        self.x_g_dot_squared = tf.tensordot(self.invar_grids[:,1,None],self.invar_grids[None,:,1],1)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        n = X.shape[0]
        m = X2.shape[0]

        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        zeros_nm =tf.zeros((n,m), dtype=tf.float64)
        zeros_mm =tf.zeros((m,m), dtype=tf.float64)
        
        Ka_X1X1  = self.Ka(X) 
        Kv_X1X1  = self.Kv(X) 
        K_X1X1   = tf.concat([tf.concat([Ka_X1X1,zeros_nn],1),tf.concat([zeros_nn,Kv_X1X1],1)],0)

        Ka_X1X2  = self.Ka(X, X2) 
        Kv_X1X2  = self.Kv(X, X2) 
        K_X1X2   = tf.concat([tf.concat([Ka_X1X2,zeros_nm],1),tf.concat([zeros_nm,Kv_X1X2],1)],0)
        
        K_X2X1   = tf.transpose(K_X1X2)
        
        Ka_X2X2  = self.Ka(X2) 
        Kv_X2X2  = self.Kv(X2) 
        K_X2X2   = tf.concat([tf.concat([Ka_X2X2,zeros_mm],1),tf.concat([zeros_mm,Kv_X2X2],1)],0)
        
        Ka_X1Xg  = self.Ka(X, self.invar_grids) 
        Kv_X1Xg  = self.Kv(X, self.invar_grids) 
        K_X1Xg = tf.concat([Ka_X1Xg, Kv_X1Xg],0)

        Ka_X2Xg =  self.Ka(X2, self.invar_grids) 
        Kv_X2Xg =  self.Kv(X2, self.invar_grids)
        K_X2Xg = tf.concat([Ka_X2Xg, Kv_X2Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g_1 = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,0]
        x_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x_g_1_stacked = tf.concat([x_g_dot_1, x_g_1],0)
        
        x_g_2 = tf.ones([m, 1], dtype=tf.float64) * self.invar_grids[:,0]
        x_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x_g_2_stacked = tf.concat([x_g_dot_2, x_g_2],0)

        
        A = tf.concat([tf.concat([K_X1X1, K_X1X2],1),tf.concat([K_X2X1, K_X2X2],1)],0) 
        B1 = tf.multiply(K_X1Xg, x_g_1_stacked)
        B2 = tf.multiply(K_X2Xg, x_g_2_stacked)
        B = tf.concat([B1, B2], 0)
        C = tf.transpose(B)
        D = tf.multiply(self.x_g_dot_squared, Ka_XgXg) + tf.multiply(self.x_g_squared, Kv_XgXg) 
        
        return (A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), C, 1))[:2*n, 2*n:]

    def K_diag(self, X):
        n = X.shape[0]
        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        
        Ka_X  = self.Ka(X) 
        Kv_X  = self.Kv(X) 
        K_X   = tf.concat([tf.concat([Ka_X,zeros_nn],1),tf.concat([zeros_nn,Kv_X],1)],0)
        
        Ka_Xg  = self.Ka(X, self.invar_grids) 
        Kv_Xg  = self.Kv(X, self.invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,0]
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x_g_stacked = tf.concat([x_g_dot, x_g],0)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(self.x_g_dot_squared, Ka_XgXg) + tf.multiply(self.x_g_squared, Kv_XgXg)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

class Pendulum_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invariance_range, invar_density, jitter_size):
        super().__init__(active_dims=[0, 1])
        self.jitter = gpflow.kernels.White(jitter_size)
        self.RBFa = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.RBFv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Ka =  self.RBFa + self.jitter
        self.Kv =  self.RBFv + self.jitter
        invariance_xs = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_vs = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_xx, invariance_vv = tf.meshgrid(invariance_xs, invariance_vs)
        self.invar_grids = tf.stack([tf.reshape(invariance_xx,[-1]), tf.reshape(invariance_vv,[-1])], axis=1)
        self.x_g_squared = tf.tensordot(tf.math.sin(self.invar_grids[:,0,None]),tf.math.sin(self.invar_grids[None,:,0]),1)
        self.x_g_dot_squared = tf.tensordot(self.invar_grids[:,1,None],self.invar_grids[None,:,1],1)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        n = X.shape[0]
        m = X2.shape[0]

        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        zeros_nm =tf.zeros((n,m), dtype=tf.float64)
        zeros_mm =tf.zeros((m,m), dtype=tf.float64)
        
        Ka_X1X1  = self.Ka(X) 
        Kv_X1X1  = self.Kv(X) 
        K_X1X1   = tf.concat([tf.concat([Ka_X1X1,zeros_nn],1),tf.concat([zeros_nn,Kv_X1X1],1)],0)

        Ka_X1X2  = self.Ka(X, X2) 
        Kv_X1X2  = self.Kv(X, X2) 
        K_X1X2   = tf.concat([tf.concat([Ka_X1X2,zeros_nm],1),tf.concat([zeros_nm,Kv_X1X2],1)],0)
        
        K_X2X1   = tf.transpose(K_X1X2)
        
        Ka_X2X2  = self.Ka(X2) 
        Kv_X2X2  = self.Kv(X2) 
        K_X2X2   = tf.concat([tf.concat([Ka_X2X2,zeros_mm],1),tf.concat([zeros_mm,Kv_X2X2],1)],0)
        
        Ka_X1Xg  = self.Ka(X, self.invar_grids) 
        Kv_X1Xg  = self.Kv(X, self.invar_grids) 
        K_X1Xg = tf.concat([Ka_X1Xg, Kv_X1Xg],0)

        Ka_X2Xg =  self.Ka(X2, self.invar_grids) 
        Kv_X2Xg =  self.Kv(X2, self.invar_grids)
        K_X2Xg = tf.concat([Ka_X2Xg, Kv_X2Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g_1 = tf.ones([n, 1], dtype=tf.float64) * tf.math.sin(self.invar_grids[:,0])
        x_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x_g_1_stacked = tf.concat([x_g_dot_1, x_g_1],0)
        
        x_g_2 = tf.ones([m, 1], dtype=tf.float64) * tf.math.sin(self.invar_grids[:,0])
        x_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x_g_2_stacked = tf.concat([x_g_dot_2, x_g_2],0)

        
        A = tf.concat([tf.concat([K_X1X1, K_X1X2],1),tf.concat([K_X2X1, K_X2X2],1)],0) 
        B1 = tf.multiply(K_X1Xg, x_g_1_stacked)
        B2 = tf.multiply(K_X2Xg, x_g_2_stacked)
        B = tf.concat([B1, B2], 0)
        C = tf.transpose(B)
        D = tf.multiply(self.x_g_dot_squared, Ka_XgXg) + tf.multiply(self.x_g_squared, Kv_XgXg) 
        
        return (A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), C, 1))[:2*n, 2*n:]

    def K_diag(self, X):
        n = X.shape[0]
        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        
        Ka_X  = self.Ka(X) 
        Kv_X  = self.Kv(X) 
        K_X   = tf.concat([tf.concat([Ka_X,zeros_nn],1),tf.concat([zeros_nn,Kv_X],1)],0)
        
        Ka_Xg  = self.Ka(X, self.invar_grids) 
        Kv_Xg  = self.Kv(X, self.invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * tf.math.sin(self.invar_grids[:,0])
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x_g_stacked = tf.concat([x_g_dot, x_g],0)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(self.x_g_dot_squared, Ka_XgXg) + tf.multiply(self.x_g_squared, Kv_XgXg)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

class zero_mean(gpflow.mean_functions.Constant):
    def __init__(self, output_dim: int = 1) -> None:
        gpflow.mean_functions.Constant.__init__(self)
        self.output_dim = output_dim
        del self.c
    def __call__(self, X) -> tf.Tensor:
        return tf.zeros((X.shape[0]*self.output_dim, 1), dtype=X.dtype)

class damping_pendulum_mean(gpflow.mean_functions.MeanFunction):
    def __init__(self, kernel, fixed, gamma, length):
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.invar_grids = kernel.invar_grids
        self.Ka = kernel.Ka
        self.Kv = kernel.Kv
        self.fixed = fixed
        self.length = length
        self.gamma = gamma
        if self.fixed:
            self.epsilon = gpflow.Parameter(0.01, transform =tfp.bijectors.Sigmoid(to_default_float(1e-6), to_default_float(1.)))

    def __call__(self, X) -> tf.Tensor:
        n = X.shape[0]
        Ka_Xg  = self.Ka(X, self.invar_grids) 
        Kv_Xg  = self.Kv(X, self.invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * tf.math.sin(self.invar_grids[:,0])
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x_g_stacked = tf.concat([x_g_dot, x_g],0)
        
        x_g_squared = tf.tensordot(tf.math.sin(self.invar_grids[:,0,None]),tf.math.sin(self.invar_grids[None,:,0]),1)
        x_g_dot_squared = tf.tensordot(self.invar_grids[:,1,None],self.invar_grids[None,:,1],1)
        B = tf.multiply(K_Xg, x_g_stacked)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg)
        if self.fixed:
            return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), -self.epsilon*tf.ones((self.invar_grids.shape[0], 1), dtype=tf.float64),1) 
        else:
            return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), -2*to_default_float(self.gamma*self.length)*tf.math.square(self.invar_grids[:,1,None]),1) 

class damping_SHM_mean(gpflow.mean_functions.MeanFunction):
    def __init__(self, kernel, fixed, gamma, mass):
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.invar_grids = kernel.invar_grids
        self.Ka = kernel.Ka
        self.Kv = kernel.Kv
        self.gamma = gamma
        self.mass = mass
        self.fixed = fixed
        if self.fixed:
            self.epsilon = gpflow.Parameter(0.01, transform =tfp.bijectors.Sigmoid(to_default_float(1e-6), to_default_float(1.)))

    def __call__(self, X) -> tf.Tensor:
        n = X.shape[0]
        Ka_Xg  = self.Ka(X, self.invar_grids) 
        Kv_Xg  = self.Kv(X, self.invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,0]
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.invar_grids[:,1]
        x_g_stacked = tf.concat([x_g_dot, x_g],0)
        
        x_g_squared = tf.tensordot(self.invar_grids[:,0,None],self.invar_grids[None,:,0],1)
        x_g_dot_squared = tf.tensordot(self.invar_grids[:,1,None],self.invar_grids[None,:,1],1)
        B = tf.multiply(K_Xg, x_g_stacked)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg)
        if self.fixed:
            return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), -self.epsilon*tf.ones((self.invar_grids.shape[0], 1), dtype=tf.float64),1) 
        else:
            return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), -2*to_default_float(self.gamma*self.mass)*tf.math.square(self.invar_grids[:,1,None]),1) 

def degree_of_freedom(kernel, test_points):
    K = kernel(test_points)
    return tf.linalg.trace(tf.tensordot(K, tf.linalg.inv(K+1e-6*tf.eye(K.shape[0], dtype=tf.float64)), 1)).numpy()

def get_SHM_data():
    t = tf.linspace(0, 30, 30)
    x = tf.math.sin(t)
    v = tf.math.cos(t)
    X1 = tf.concat([x[:,None], v[:,None]], axis=-1)
    X2 = 2*X1
    X1 += tf.random.normal((X1.shape), 0, 0.1, dtype=tf.float64)
    X2 += tf.random.normal((X2.shape), 0, 0.1, dtype=tf.float64)
    Y1 = (X1[2:,:]-X1[:-2, :])/(2) 
    Y2 = (X2[2:,:]-X2[:-2, :])/(2) 
    X1 = X1[1:-1, :]
    X2 = X2[1:-1, :]
    X = tf.concat([X1, X2], axis=0)
    Y = tf.concat([Y1, Y2], axis=0) 
    return (X, Y)

def get_damped_SHM_data(gamma):
    m = 1
    k = 1
    w = np.sqrt(k/m-gamma**2)
    t = np.linspace(0, 30, 30)
    x = np.sin(w*t)*np.exp(-gamma*t)
    v = np.exp(-gamma*t)*(w*np.cos(w*t)-gamma*np.sin(w*t))
    X1 = tf.concat([x[:,None], v[:,None]], axis=-1)
    X2 = 2*X1
    X1 += tf.random.normal((X1.shape), 0, 0.1, dtype=tf.float64)
    X2 += tf.random.normal((X2.shape), 0, 0.1, dtype=tf.float64)
    Y1 = (X1[2:,:]-X1[:-2, :])/(2) 
    Y2 = (X2[2:,:]-X2[:-2, :])/(2) 
    X1 = X1[1:-1, :]
    X2 = X2[1:-1, :]
    X = tf.concat([X1, X2], axis=0)
    Y = tf.concat([Y1, Y2], axis=0)
    return (X, Y)

def get_pendulum_data():
    dt = 0.01
    sample = int(1/dt)
    t = np.linspace(0, 30, int(30/dt))
    g = 1
    l = 1
    def f(t, r):
        theta = r[0]
        omega = r[1]
        return np.array([omega, -g / l * np.sin(theta)])
    results = odeint(f, [np.radians(90), 0], t, tfirst=True)
    results2 = odeint(f, [np.radians(150), 0], t, tfirst=True)
    x1 = results[0::sample,0]
    v1 = results[0::sample,1]
    x2 = results2[0::sample,0]
    v2 = results2[0::sample,1]
    t = t[0::sample]
    X_1 = tf.concat([x1[:,None], v1[:,None]], axis=-1)
    X_2 = tf.concat([x2[:,None], v2[:,None]], axis=-1)
    X_1 += tf.random.normal((X_1.shape), 0, 0.1, dtype=tf.float64)
    X_2 += tf.random.normal((X_2.shape), 0, 0.1, dtype=tf.float64)
    Y_1 = (X_1[2:,:]-X_1[:-2, :])/(2) 
    Y_2 = (X_2[2:,:]-X_2[:-2, :])/(2) 
    X_1 = X_1[1:-1, :]
    X_2 = X_2[1:-1, :]
    X = tf.concat([X_1,X_2], axis=0)
    Y = tf.concat([Y_1,Y_2], axis=0)
    return (X, Y)

def get_damped_pendulum_data(gamma):
    dt = 0.01
    sample = int(1/dt)
    t = np.linspace(0, 30, int(30/dt))
    g = 1
    l = 1
    def f(t, r):
        theta = r[0]
        omega = r[1]
        return np.array([omega, -g / l * np.sin(theta)-2*gamma*omega])
    results = odeint(f, [np.radians(90), 0], t, tfirst=True)
    results2 = odeint(f, [np.radians(150), 0], t, tfirst=True)
    x1 = results[0::sample,0]
    v1 = results[0::sample,1]
    x2 = results2[0::sample,0]
    v2 = results2[0::sample,1]
    t = t[0::sample]
    X_1 = tf.concat([x1[:,None], v1[:,None]], axis=-1)
    X_2 = tf.concat([x2[:,None], v2[:,None]], axis=-1)
    X_1 += tf.random.normal((X_1.shape), 0, 0.1, dtype=tf.float64)
    X_2 += tf.random.normal((X_2.shape), 0, 0.1, dtype=tf.float64)
    Y_1 = (X_1[2:,:]-X_1[:-2, :])/(2) 
    Y_2 = (X_2[2:,:]-X_2[:-2, :])/(2) 
    X_1 = X_1[1:-1, :]
    X_2 = X_2[1:-1, :]
    X = tf.concat([X_1,X_2], axis=0)
    Y = tf.concat([Y_1,Y_2], axis=0)
    return (X, Y)

def get_test_points():
    test_range = 3 
    test_density = 40
    test_xs = tf.linspace(-test_range,test_range,test_density)
    test_vs = tf.linspace(-test_range,test_range,test_density)
    test_xx, test_vv = tf.meshgrid(test_xs, test_vs)
    test_points = tf.stack([tf.reshape(test_xx,[-1]), tf.reshape(test_vv,[-1])], axis=1)
    return test_points

def get_MOI(jitter_size):
    moi = MOI(jitter_size)
    set_trainable(moi.jitter.variance, False)
    moi.RBFa.variance = gpflow.Parameter(moi.RBFa.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    moi.RBFv.variance = gpflow.Parameter(moi.RBFv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    moi.RBFa.lengthscales = gpflow.Parameter(moi.RBFa.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    moi.RBFv.lengthscales = gpflow.Parameter(moi.RBFv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    return moi
    
def get_SHM_Invariance(invar_range, invar_density, jitter_size):
    invariance_kernel = SHM_Invariance(invar_range, invar_density, jitter_size)
    set_trainable(invariance_kernel.jitter.variance, False)
    invariance_kernel.RBFa.variance = gpflow.Parameter(invariance_kernel.RBFa.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    invariance_kernel.RBFv.variance = gpflow.Parameter(invariance_kernel.RBFv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    invariance_kernel.RBFa.lengthscales = gpflow.Parameter(invariance_kernel.RBFa.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    invariance_kernel.RBFv.lengthscales = gpflow.Parameter(invariance_kernel.RBFv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    return invariance_kernel

def get_Pendulum_Invariance(invar_range, invar_density, jitter_size):
    invariance_kernel = Pendulum_Invariance(invar_range, invar_density, jitter_size)
    set_trainable(invariance_kernel.jitter.variance, False)
    invariance_kernel.RBFa.variance = gpflow.Parameter(invariance_kernel.RBFa.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    invariance_kernel.RBFv.variance = gpflow.Parameter(invariance_kernel.RBFv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    invariance_kernel.RBFa.lengthscales = gpflow.Parameter(invariance_kernel.RBFa.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    invariance_kernel.RBFv.lengthscales = gpflow.Parameter(invariance_kernel.RBFv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
    return invariance_kernel

def get_GPR_model(kernel, mean_function, data, test_points):
    X, Y = data
    m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,1,None],Y[:,0,None]],1)),(Y.shape[0]*2,1))), kernel=kernel, mean_function=mean_function)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
    pred, var = m.predict_f(test_points)
    return (m, pred, var)

                

# %%
