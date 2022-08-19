
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable

class SHMEpsilonMean(gpflow.mean_functions.MeanFunction):
    def __init__(self, kernel):
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.invar_grids = kernel.invar_grids
        self.Ka = kernel.Ka
        self.Kv = kernel.Kv
        self.jitter = kernel.jitter
        self.x_g_squared = kernel.x_g_squared
        self.x_g_dot_squared = kernel.x_g_dot_squared
        self.gamma = gpflow.Parameter(1, transform =positive())#tfp.bijectors.Sigmoid(to_default_float(), to_default_float(3.)))
        self.epsilon = kernel.epsilon#gpflow.Parameter(0.01, transform =tfp.bijectors.Sigmoid(to_default_float(self.jitter), to_default_float(1.)))

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
        
        B = tf.multiply(K_Xg, x_g_stacked)
        D = tf.multiply(self.x_g_dot_squared, Ka_XgXg) + tf.multiply(self.x_g_squared, Kv_XgXg)
        D += self.epsilon*tf.eye(D.shape[0], dtype=tf.float64)
        return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), -self.gamma*tf.ones((self.invar_grids.shape[0], 1), dtype=tf.float64),1) 

class PendulumEpsilonMean(gpflow.mean_functions.MeanFunction):
    def __init__(self, kernel):
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.invar_grids = kernel.invar_grids
        self.Ka = kernel.Ka
        self.Kv = kernel.Kv
        self.jitter = kernel.jitter
        self.x_g_squared = kernel.x_g_squared
        self.x_g_dot_squared = kernel.x_g_dot_squared
        self.gamma = gpflow.Parameter(1, transform =tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.)))
        self.epsilon = gpflow.Parameter(0.01, transform =tfp.bijectors.Sigmoid(to_default_float(self.jitter), to_default_float(1.)))

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
        
        B = tf.multiply(K_Xg, x_g_stacked)
        D = tf.multiply(self.x_g_dot_squared, Ka_XgXg) + tf.multiply(self.x_g_squared, Kv_XgXg)
        D += self.epsilon*tf.eye(D.shape[0], dtype=tf.float64)
        return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), -self.gamma*tf.ones((self.invar_grids.shape[0], 1), dtype=tf.float64),1) 


class PolynomialEpsilonMean(gpflow.mean_functions.MeanFunction):
    def __init__(self, kernel):
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.invar_grids = kernel.invar_grids
        self.Ka = kernel.Ka
        self.Kv = kernel.Kv
        self.inv_f = kernel.inv_f
        self.inv_g = kernel.inv_g
        self.jitter = kernel.jitter
        self.x_g_squared = kernel.x_g_squared
        self.x_g_dot_squared = kernel.x_g_dot_squared
        self.gamma = gpflow.Parameter(1, transform =tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.)))
        self.epsilon = gpflow.Parameter(0.01, transform =tfp.bijectors.Sigmoid(to_default_float(self.jitter), to_default_float(1.)))

    def __call__(self, X) -> tf.Tensor:
        n = X.shape[0]
        Ka_Xg  = self.Ka(X, self.invar_grids) 
        Kv_Xg  = self.Kv(X, self.invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(self.invar_grids[:,0, None]))
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(self.invar_grids[:,1, None]))
        x_g_stacked = tf.concat([x_g_dot, x_g],0)
        
        B = tf.multiply(K_Xg, x_g_stacked)
        D = tf.multiply(self.x_g_dot_squared, Ka_XgXg) + tf.multiply(self.x_g_squared, Kv_XgXg)
        D += self.epsilon*tf.eye(D.shape[0], dtype=tf.float64)
        return tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), -self.gamma*tf.ones((self.invar_grids.shape[0], 1), dtype=tf.float64),1) 
