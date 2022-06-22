import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable

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
        self.jitter = kernel.jitter
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
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        if self.fixed:
            return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), self.epsilon*tf.random.normal((self.invar_grids.shape[0], 1), dtype=tf.float64),1) 
        else:
            return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), -2*to_default_float(self.gamma*self.length)*tf.math.square(self.invar_grids[:,1,None]),1) 

class damping_SHM_mean(gpflow.mean_functions.MeanFunction):
    def __init__(self, kernel, fixed, gamma, mass):
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.invar_grids = kernel.invar_grids
        self.Ka = kernel.Ka
        self.Kv = kernel.Kv
        self.jitter = kernel.jitter
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
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        if self.fixed:
            return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), self.epsilon*tf.random.normal((self.invar_grids.shape[0], 1), dtype=tf.float64),1) 
        else:
            return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), -2*to_default_float(self.gamma*self.mass)*tf.math.square(self.invar_grids[:,1,None]),1) 