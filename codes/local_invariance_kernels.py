import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable

class SHM_Local_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood, jitter_size):
        super().__init__(active_dims=[0, 1])
        self.Ka = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Kv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.jitter = jitter_size
        self.invar_neighbourhood = invar_neighbourhood

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

        local_X1_invar_grids1 = X+self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X1_invar_grids2 = X+self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_X1_invar_grids3 = X-self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X1_invar_grids4 = X-self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_X2_invar_grids1 = X+self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X2_invar_grids2 = X+self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_X2_invar_grids3 = X-self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X2_invar_grids4 = X-self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_invar_grids = tf.concat([local_X1_invar_grids1, local_X1_invar_grids2, local_X1_invar_grids3, local_X1_invar_grids4,local_X2_invar_grids1, local_X2_invar_grids2, local_X2_invar_grids3, local_X2_invar_grids4 ],0)

        Ka_X1Xg  = self.Ka(X, local_invar_grids) 
        Kv_X1Xg  = self.Kv(X, local_invar_grids) 
        K_X1Xg = tf.concat([Ka_X1Xg, Kv_X1Xg],0)

        Ka_X2Xg =  self.Ka(X2, local_invar_grids) 
        Kv_X2Xg =  self.Kv(X2, local_invar_grids)
        K_X2Xg = tf.concat([Ka_X2Xg, Kv_X2Xg],0)

        Ka_XgXg = self.Ka(local_invar_grids) 
        Kv_XgXg = self.Kv(local_invar_grids) 
        
        x_g_1 = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,0]
        x_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,1]
        x_g_1_stacked = tf.concat([x_g_dot_1, x_g_1],0)
        
        x_g_2 = tf.ones([m, 1], dtype=tf.float64) * local_invar_grids[:,0]
        x_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * local_invar_grids[:,1]
        x_g_2_stacked = tf.concat([x_g_dot_2, x_g_2],0)
        
        x_g_squared = tf.tensordot(local_invar_grids[:,0,None],local_invar_grids[None,:,0],1)
        x_g_dot_squared = tf.tensordot(local_invar_grids[:,1,None],local_invar_grids[None,:,1],1)
        
        A = tf.concat([tf.concat([K_X1X1, K_X1X2],1),tf.concat([K_X2X1, K_X2X2],1)],0) 
        B1 = tf.multiply(K_X1Xg, x_g_1_stacked)
        B2 = tf.multiply(K_X2Xg, x_g_2_stacked)
        B = tf.concat([B1, B2], 0)
        C = tf.transpose(B)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg) 
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        
        return (A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), C, 1))[:2*n, 2*n:]

    def K_diag(self, X):
        n = X.shape[0]
        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        
        Ka_X  = self.Ka(X) 
        Kv_X  = self.Kv(X) 
        K_X   = tf.concat([tf.concat([Ka_X,zeros_nn],1),tf.concat([zeros_nn,Kv_X],1)],0)

        local_X_invar_grids1 = X+self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X_invar_grids2 = X+self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_X_invar_grids3 = X-self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X_invar_grids4 = X-self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_invar_grids = tf.concat([local_X_invar_grids1, local_X_invar_grids2, local_X_invar_grids3, local_X_invar_grids4],0)
        
        Ka_Xg  = self.Ka(X, local_invar_grids) 
        Kv_Xg  = self.Kv(X, local_invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(local_invar_grids) 
        Kv_XgXg = self.Kv(local_invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,0]
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,1]
        x_g_stacked = tf.concat([x_g_dot, x_g],0)

        x_g_squared = tf.tensordot(local_invar_grids[:,0,None],local_invar_grids[None,:,0],1)
        x_g_dot_squared = tf.tensordot(local_invar_grids[:,1,None],local_invar_grids[None,:,1],1)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg)
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)

        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))


class Pendulum_Local_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood, jitter_size):
        super().__init__(active_dims=[0, 1])
        self.invar_neighbourhood = invar_neighbourhood
        self.jitter = jitter_size
        self.Ka = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Kv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 

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

        local_X1_invar_grids1 = X+self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X1_invar_grids2 = X+self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_X1_invar_grids3 = X-self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X1_invar_grids4 = X-self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_X2_invar_grids1 = X+self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X2_invar_grids2 = X+self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_X2_invar_grids3 = X-self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X2_invar_grids4 = X-self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_invar_grids = tf.concat([local_X1_invar_grids1, local_X1_invar_grids2, local_X1_invar_grids3, local_X1_invar_grids4,local_X2_invar_grids1, local_X2_invar_grids2, local_X2_invar_grids3, local_X2_invar_grids4 ],0)
        
        Ka_X1Xg  = self.Ka(X, local_invar_grids) 
        Kv_X1Xg  = self.Kv(X, local_invar_grids) 
        K_X1Xg = tf.concat([Ka_X1Xg, Kv_X1Xg],0)

        Ka_X2Xg =  self.Ka(X2, local_invar_grids) 
        Kv_X2Xg =  self.Kv(X2, local_invar_grids)
        K_X2Xg = tf.concat([Ka_X2Xg, Kv_X2Xg],0)

        Ka_XgXg = self.Ka(local_invar_grids) 
        Kv_XgXg = self.Kv(local_invar_grids) 
        
        x_g_1 = tf.ones([n, 1], dtype=tf.float64) * tf.math.sin(local_invar_grids[:,0])
        x_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,1]
        x_g_1_stacked = tf.concat([x_g_dot_1, x_g_1],0)
        
        x_g_2 = tf.ones([m, 1], dtype=tf.float64) * tf.math.sin(local_invar_grids[:,0])
        x_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * local_invar_grids[:,1]
        x_g_2_stacked = tf.concat([x_g_dot_2, x_g_2],0)

        x_g_squared = tf.tensordot(tf.math.sin(local_invar_grids[:,0,None]),tf.math.sin(local_invar_grids[None,:,0]),1)
        x_g_dot_squared = tf.tensordot(local_invar_grids[:,1,None],local_invar_grids[None,:,1],1)

        A = tf.concat([tf.concat([K_X1X1, K_X1X2],1),tf.concat([K_X2X1, K_X2X2],1)],0) 
        B1 = tf.multiply(K_X1Xg, x_g_1_stacked)
        B2 = tf.multiply(K_X2Xg, x_g_2_stacked)
        B = tf.concat([B1, B2], 0)
        C = tf.transpose(B)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg) 
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        
        return (A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), C, 1))[:2*n, 2*n:]

    def K_diag(self, X):
        n = X.shape[0]
        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        
        Ka_X  = self.Ka(X) 
        Kv_X  = self.Kv(X) 
        K_X   = tf.concat([tf.concat([Ka_X,zeros_nn],1),tf.concat([zeros_nn,Kv_X],1)],0)

        local_X_invar_grids1 = X+self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X_invar_grids2 = X+self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_X_invar_grids3 = X-self.invar_neighbourhood*tf.ones((X.shape), dtype=tf.float64)
        local_X_invar_grids4 = X-self.invar_neighbourhood*tf.concat([tf.ones((n, 1), dtype=tf.float64), -tf.ones((n, 1), dtype=tf.float64)], 1)
        local_invar_grids = tf.concat([local_X_invar_grids1, local_X_invar_grids2, local_X_invar_grids3, local_X_invar_grids4],0)
        
        Ka_Xg  = self.Ka(X, local_invar_grids) 
        Kv_Xg  = self.Kv(X, local_invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(local_invar_grids) 
        Kv_XgXg = self.Kv(local_invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * tf.math.sin(local_invar_grids[:,0])
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,1]
        x_g_stacked = tf.concat([x_g_dot, x_g],0)

        x_g_squared = tf.tensordot(tf.math.sin(local_invar_grids[:,0,None]),tf.math.sin(local_invar_grids[None,:,0]),1)
        x_g_dot_squared = tf.tensordot(local_invar_grids[:,1,None],local_invar_grids[None,:,1],1)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg)
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

def get_SHM_Local_Invariance(invar_neighbourhood, jitter_size):
    invariance_kernel = SHM_Local_Invariance(invar_neighbourhood, jitter_size)
    invariance_kernel.Ka.variance = gpflow.Parameter(invariance_kernel.Ka.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.variance = gpflow.Parameter(invariance_kernel.Kv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka.lengthscales = gpflow.Parameter(invariance_kernel.Ka.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.lengthscales = gpflow.Parameter(invariance_kernel.Kv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    return invariance_kernel

def get_Pendulum_Local_Invariance(invar_neighbourhood, jitter_size):
    invariance_kernel = Pendulum_Local_Invariance(invar_neighbourhood, jitter_size)
    invariance_kernel.Ka.variance = gpflow.Parameter(invariance_kernel.Ka.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.variance = gpflow.Parameter(invariance_kernel.Kv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka.lengthscales = gpflow.Parameter(invariance_kernel.Ka.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.lengthscales = gpflow.Parameter(invariance_kernel.Kv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    return invariance_kernel