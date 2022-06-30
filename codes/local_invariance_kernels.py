import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from scipy.special import comb
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from itertools import combinations_with_replacement

class SHM_Local_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood, n_neighbours, jitter_size):
        super().__init__(active_dims=[0, 1])
        self.Ka = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Kv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.jitter = jitter_size
        self.n_neighbours = n_neighbours
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

        local_X1_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_X1_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X1_invar_grids.shape), dtype=tf.float64)
        local_X2_invar_grids = tf.repeat(X2, self.n_neighbours, 0) 
        local_X2_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X2_invar_grids.shape), dtype=tf.float64)
        
        local_invar_grids = tf.concat([local_X1_invar_grids, local_X2_invar_grids],0)

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

        local_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_invar_grids += self.invar_neighbourhood*tf.random.normal((local_invar_grids.shape), dtype=tf.float64)
        
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

class SHM2D_Local_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood, n_neighbours, jitter_size):
        super().__init__(active_dims=[0, 1, 2, 3])
        self.Ka1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Ka2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.jitter = jitter_size
        self.n_neighbours = n_neighbours
        self.invar_neighbourhood = invar_neighbourhood


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

        local_X1_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_X1_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X1_invar_grids.shape), dtype=tf.float64)
        local_X2_invar_grids = tf.repeat(X2, self.n_neighbours, 0) 
        local_X2_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X2_invar_grids.shape), dtype=tf.float64)
        
        local_invar_grids = tf.concat([local_X1_invar_grids, local_X2_invar_grids],0)
        
        Ka1_X1Xg  = self.Ka1(X, local_invar_grids) 
        Ka2_X1Xg  = self.Ka2(X, local_invar_grids) 
        Kv1_X1Xg  = self.Kv1(X, local_invar_grids) 
        Kv2_X1Xg  = self.Kv2(X, local_invar_grids) 
        K_X1Xg = tf.concat([Ka1_X1Xg, Ka2_X1Xg, Kv1_X1Xg, Kv2_X1Xg],0)

        Ka1_X2Xg =  self.Ka1(X2, local_invar_grids) 
        Ka2_X2Xg =  self.Ka2(X2, local_invar_grids) 
        Kv1_X2Xg =  self.Kv1(X2, local_invar_grids)
        Kv2_X2Xg =  self.Kv2(X2, local_invar_grids)
        K_X2Xg = tf.concat([Ka1_X2Xg, Ka2_X2Xg, Kv1_X2Xg, Kv2_X2Xg],0)

        Ka1_XgXg = self.Ka1(local_invar_grids) 
        Ka2_XgXg = self.Ka2(local_invar_grids) 
        Kv1_XgXg = self.Kv1(local_invar_grids) 
        Kv2_XgXg = self.Kv2(local_invar_grids) 
        
        x1_g_1 = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,0]
        x2_g_1 = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,1]
        x1_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,2]
        x2_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,3]
        x_g_1_stacked = tf.concat([x1_g_dot_1, x2_g_dot_1, x1_g_1, x2_g_1],0)
        
        x1_g_2 = tf.ones([m, 1], dtype=tf.float64) * local_invar_grids[:,0]
        x2_g_2 = tf.ones([m, 1], dtype=tf.float64) * local_invar_grids[:,1]
        x1_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * local_invar_grids[:,2]
        x2_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * local_invar_grids[:,3]
        x_g_2_stacked = tf.concat([x1_g_dot_2, x2_g_dot_2, x1_g_2, x2_g_2],0)

        x1_g_squared = tf.tensordot(local_invar_grids[:,0,None],local_invar_grids[None,:,0],1)
        x2_g_squared = tf.tensordot(local_invar_grids[:,1,None],local_invar_grids[None,:,1],1)
        x1_g_dot_squared = tf.tensordot(local_invar_grids[:,2,None],local_invar_grids[None,:,2],1)
        x2_g_dot_squared = tf.tensordot(local_invar_grids[:,3,None],local_invar_grids[None,:,3],1)

        
        A = tf.concat([tf.concat([K_X1X1, K_X1X2],1),tf.concat([K_X2X1, K_X2X2],1)],0) 
        B1 = tf.multiply(K_X1Xg, x_g_1_stacked)
        B2 = tf.multiply(K_X2Xg, x_g_2_stacked)
        B = tf.concat([B1, B2], 0)
        C = tf.transpose(B)
        D = tf.multiply(x1_g_dot_squared, Ka1_XgXg) + tf.multiply(x2_g_dot_squared, Ka2_XgXg) + tf.multiply(x1_g_squared, Kv1_XgXg) + tf.multiply(x2_g_squared, Kv2_XgXg) 
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

        local_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_invar_grids += self.invar_neighbourhood*tf.random.normal((local_invar_grids.shape), dtype=tf.float64)
        
        Ka1_Xg  = self.Ka1(X, local_invar_grids) 
        Ka2_Xg  = self.Ka2(X, local_invar_grids) 
        Kv1_Xg  = self.Kv1(X, local_invar_grids) 
        Kv2_Xg  = self.Kv2(X, local_invar_grids) 
        K_Xg = tf.concat([Ka1_Xg, Ka2_Xg, Kv1_Xg, Kv2_Xg],0)

        Ka1_XgXg = self.Ka1(local_invar_grids) 
        Ka2_XgXg = self.Ka2(local_invar_grids) 
        Kv1_XgXg = self.Kv1(local_invar_grids) 
        Kv2_XgXg = self.Kv2(local_invar_grids) 
        
        x1_g = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,0]
        x2_g = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,1]
        x1_g_dot = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,2]
        x2_g_dot = tf.ones([n, 1], dtype=tf.float64) * local_invar_grids[:,3]
        x_g_stacked = tf.concat([x1_g_dot, x2_g_dot, x1_g, x2_g],0)

        x1_g_squared = tf.tensordot(local_invar_grids[:,0,None],local_invar_grids[None,:,0],1)
        x2_g_squared = tf.tensordot(local_invar_grids[:,1,None],local_invar_grids[None,:,1],1)
        x1_g_dot_squared = tf.tensordot(local_invar_grids[:,2,None],local_invar_grids[None,:,2],1)
        x2_g_dot_squared = tf.tensordot(local_invar_grids[:,3,None],local_invar_grids[None,:,3],1)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(x1_g_dot_squared, Ka1_XgXg) + tf.multiply(x2_g_dot_squared, Ka2_XgXg) + tf.multiply(x1_g_squared, Kv1_XgXg) + tf.multiply(x2_g_squared, Kv2_XgXg) 
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

class Pendulum_Local_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood, n_neighbours, jitter_size):
        super().__init__(active_dims=[0, 1])
        self.n_neighbours = n_neighbours
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

        local_X1_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_X1_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X1_invar_grids.shape), dtype=tf.float64)
        local_X2_invar_grids = tf.repeat(X2, self.n_neighbours, 0) 
        local_X2_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X2_invar_grids.shape), dtype=tf.float64)
        
        local_invar_grids = tf.concat([local_X1_invar_grids, local_X2_invar_grids],0)

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

        local_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_invar_grids += self.invar_neighbourhood*tf.random.normal((local_invar_grids.shape), dtype=tf.float64)

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

class Double_Pendulum_Local_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood, n_neighbours, jitter_size):
        super().__init__(active_dims=[0, 1, 2, 3])
        self.Ka1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Ka2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.jitter = jitter_size
        self.invar_neighbourhood = invar_neighbourhood
        self.n_neighbours = n_neighbours

    def inv_f1(self, X):
        return 2*X[:,2] + X[:,3]*tf.math.cos(X[:,0]-X[:,1])
    def inv_f2(self, X):
        return X[:,3]+X[:,2]*tf.math.cos(X[:,0]-X[:,1])
    def inv_g1(self, X):
        return 2*tf.math.sin(X[:,0])-X[:,2]*X[:,3]*tf.math.sin(X[:,0]-X[:,1])
    def inv_g2(self, X):
        return tf.math.sin(X[:,1])+X[:,2]*X[:,3]*tf.math.sin(X[:,0]-X[:,1])

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

        local_X1_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_X1_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X1_invar_grids.shape), dtype=tf.float64)
        local_X2_invar_grids = tf.repeat(X2, self.n_neighbours, 0) 
        local_X2_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X2_invar_grids.shape), dtype=tf.float64)
        
        local_invar_grids = tf.concat([local_X1_invar_grids, local_X2_invar_grids],0)
        
        Ka1_X1Xg  = self.Ka1(X, local_invar_grids) 
        Ka2_X1Xg  = self.Ka2(X, local_invar_grids) 
        Kv1_X1Xg  = self.Kv1(X, local_invar_grids) 
        Kv2_X1Xg  = self.Kv2(X, local_invar_grids) 
        K_X1Xg = tf.concat([Ka1_X1Xg, Ka2_X1Xg, Kv1_X1Xg, Kv2_X1Xg],0)

        Ka1_X2Xg =  self.Ka1(X2, local_invar_grids) 
        Ka2_X2Xg =  self.Ka2(X2, local_invar_grids) 
        Kv1_X2Xg =  self.Kv1(X2, local_invar_grids)
        Kv2_X2Xg =  self.Kv2(X2, local_invar_grids)
        K_X2Xg = tf.concat([Ka1_X2Xg, Ka2_X2Xg, Kv1_X2Xg, Kv2_X2Xg],0)

        Ka1_XgXg = self.Ka1(local_invar_grids) 
        Ka2_XgXg = self.Ka2(local_invar_grids) 
        Kv1_XgXg = self.Kv1(local_invar_grids) 
        Kv2_XgXg = self.Kv2(local_invar_grids) 
        
        x1_g_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_g1(local_invar_grids)
        x2_g_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_g2(local_invar_grids)
        x1_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_f1(local_invar_grids)
        x2_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_f2(local_invar_grids)
        x_g_1_stacked = tf.concat([x1_g_dot_1, x2_g_dot_1, x1_g_1, x2_g_1],0)

        x1_g_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_g1(local_invar_grids)
        x2_g_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_g2(local_invar_grids)
        x1_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_f1(local_invar_grids)
        x2_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_f2(local_invar_grids)
        x_g_2_stacked = tf.concat([x1_g_dot_2, x2_g_dot_2, x1_g_2, x2_g_2],0)

        x1_g_not_squared = self.inv_g1(local_invar_grids)[:,None]
        x2_g_not_squared = self.inv_g2(local_invar_grids)[:,None]
        x1_g_dot_not_squared = self.inv_f1(local_invar_grids)[:,None]
        x2_g_dot_not_squared = self.inv_f2(local_invar_grids)[:,None]
        x1_g_squared = tf.tensordot(x1_g_not_squared, tf.transpose(x1_g_not_squared), 1)
        x2_g_squared = tf.tensordot(x2_g_not_squared, tf.transpose(x2_g_not_squared), 1)
        x1_g_dot_squared = tf.tensordot(x1_g_dot_not_squared, tf.transpose(x1_g_dot_not_squared), 1)
        x2_g_dot_squared = tf.tensordot(x2_g_dot_not_squared, tf.transpose(x2_g_dot_not_squared), 1)
        
        A = tf.concat([tf.concat([K_X1X1, K_X1X2],1),tf.concat([K_X2X1, K_X2X2],1)],0) 
        B1 = tf.multiply(K_X1Xg, x_g_1_stacked)
        B2 = tf.multiply(K_X2Xg, x_g_2_stacked)
        B = tf.concat([B1, B2], 0)
        C = tf.transpose(B)
        D = tf.multiply(x1_g_dot_squared, Ka1_XgXg) + tf.multiply(x2_g_dot_squared, Ka2_XgXg) + tf.multiply(x1_g_squared, Kv1_XgXg) + tf.multiply(x2_g_squared, Kv2_XgXg) 
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

        local_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_invar_grids += self.invar_neighbourhood*tf.random.normal((local_invar_grids.shape), dtype=tf.float64)
        
        Ka1_Xg  = self.Ka1(X, local_invar_grids) 
        Ka2_Xg  = self.Ka2(X, local_invar_grids) 
        Kv1_Xg  = self.Kv1(X, local_invar_grids) 
        Kv2_Xg  = self.Kv2(X, local_invar_grids) 
        K_Xg = tf.concat([Ka1_Xg, Ka2_Xg, Kv1_Xg, Kv2_Xg],0)

        Ka1_XgXg = self.Ka1(local_invar_grids) 
        Ka2_XgXg = self.Ka2(local_invar_grids) 
        Kv1_XgXg = self.Kv1(local_invar_grids) 
        Kv2_XgXg = self.Kv2(local_invar_grids) 
        
        x1_g = tf.ones([n, 1], dtype=tf.float64) * self.inv_g1(local_invar_grids)
        x2_g = tf.ones([n, 1], dtype=tf.float64) * self.inv_g2(local_invar_grids)
        x1_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.inv_f1(local_invar_grids)
        x2_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.inv_f2(local_invar_grids)
        x_g_stacked = tf.concat([x1_g_dot, x2_g_dot, x1_g, x2_g],0)

        x1_g_not_squared = self.inv_g1(local_invar_grids)[:,None]
        x2_g_not_squared = self.inv_g2(local_invar_grids)[:,None]
        x1_g_dot_not_squared = self.inv_f1(local_invar_grids)[:,None]
        x2_g_dot_not_squared = self.inv_f2(local_invar_grids)[:,None]
        x1_g_squared = tf.tensordot(x1_g_not_squared, tf.transpose(x1_g_not_squared), 1)
        x2_g_squared = tf.tensordot(x2_g_not_squared, tf.transpose(x2_g_not_squared), 1)
        x1_g_dot_squared = tf.tensordot(x1_g_dot_not_squared, tf.transpose(x1_g_dot_not_squared), 1)
        x2_g_dot_squared = tf.tensordot(x2_g_dot_not_squared, tf.transpose(x2_g_dot_not_squared), 1)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(x1_g_dot_squared, Ka1_XgXg) + tf.multiply(x2_g_dot_squared, Ka2_XgXg) + tf.multiply(x1_g_squared, Kv1_XgXg) + tf.multiply(x2_g_squared, Kv2_XgXg) 
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

class Polynomial_Local_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood, n_neighbours, jitter_size, poly_f_d, poly_g_d):
        super().__init__(active_dims=[0, 1])
        self.poly_f_d = poly_f_d
        self.poly_g_d = poly_g_d
        self.f_poly = gpflow.Parameter(tf.Variable(0.5*tf.random.normal((self.poly_f_d, 1), dtype=tf.float64)), transform =tfp.bijectors.Sigmoid(to_default_float(-2.), to_default_float(2.)))
        self.g_poly = gpflow.Parameter(tf.Variable(0.5*tf.random.normal((self.poly_g_d, 1), dtype=tf.float64)), transform =tfp.bijectors.Sigmoid(to_default_float(-2.), to_default_float(2.)))

        self.n_neighbours = n_neighbours
        self.invar_neighbourhood = invar_neighbourhood
        self.jitter =jitter_size
        self.Ka = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Kv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 

    def inv_f(self, X):
        return tf.linalg.matmul(tf.math.pow(X, list(range(self.poly_f_d))), self.f_poly)
    def inv_g(self, X):
        return tf.linalg.matmul(tf.math.pow(X, list(range(self.poly_g_d))), self.g_poly)

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

        local_X1_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_X1_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X1_invar_grids.shape), dtype=tf.float64)
        local_X2_invar_grids = tf.repeat(X2, self.n_neighbours, 0) 
        local_X2_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X2_invar_grids.shape), dtype=tf.float64)
        
        local_invar_grids = tf.concat([local_X1_invar_grids, local_X2_invar_grids],0)

        Ka_X1Xg  = self.Ka(X, local_invar_grids) 
        Kv_X1Xg  = self.Kv(X, local_invar_grids) 
        K_X1Xg = tf.concat([Ka_X1Xg, Kv_X1Xg],0)

        Ka_X2Xg =  self.Ka(X2, local_invar_grids) 
        Kv_X2Xg =  self.Kv(X2, local_invar_grids)
        K_X2Xg = tf.concat([Ka_X2Xg, Kv_X2Xg],0)

        Ka_XgXg = self.Ka(local_invar_grids) 
        Kv_XgXg = self.Kv(local_invar_grids) 
        
        x_g_1 = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(local_invar_grids[:,0, None]))
        x_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(local_invar_grids[:,1, None]))
        x_g_1_stacked = tf.concat([x_g_dot_1, x_g_1],0) 
        
        x_g_2 = tf.ones([m, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(local_invar_grids[:,0, None]))
        x_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(local_invar_grids[:,1, None]))
        x_g_2_stacked = tf.concat([x_g_dot_2, x_g_2],0) 

        x_g_squared = tf.tensordot(self.inv_g(local_invar_grids[:,0,None]),tf.transpose(self.inv_g(local_invar_grids[:,0, None])),1)
        x_g_dot_squared = tf.tensordot(self.inv_f(local_invar_grids[:,1,None]),tf.transpose(self.inv_f(local_invar_grids[:,1, None])),1)
        
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

        local_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_invar_grids += self.invar_neighbourhood*tf.random.normal((local_invar_grids.shape), dtype=tf.float64)

        Ka_Xg  = self.Ka(X, local_invar_grids) 
        Kv_Xg  = self.Kv(X, local_invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(local_invar_grids) 
        Kv_XgXg = self.Kv(local_invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(local_invar_grids[:,0, None]))
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(local_invar_grids[:,1, None]))
        x_g_stacked = tf.concat([x_g_dot, x_g],0)
        
        x_g_squared = tf.tensordot(self.inv_g(local_invar_grids[:,0,None]),tf.transpose(self.inv_g(local_invar_grids[:,0, None])),1)
        x_g_dot_squared = tf.tensordot(self.inv_f(local_invar_grids[:,1,None]),tf.transpose(self.inv_f(local_invar_grids[:,1, None])),1)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg)
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

class Polynomial_2D_Local_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood, n_neighbours, jitter_size, poly_d):
        super().__init__(active_dims=[0, 1, 2, 3])
        #poly_d = [d, d, d, d]
        self.Ka1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Ka2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.poly_d = poly_d
        self.f1_poly = gpflow.Parameter(tf.Variable(0.5*tf.random.normal((self.number_of_coefficients(self.poly_d[0]),1), dtype=tf.float64)), transform =tfp.bijectors.Sigmoid(to_default_float(-2.), to_default_float(2.)))
        self.f2_poly = gpflow.Parameter(tf.Variable(0.5*tf.random.normal((self.number_of_coefficients(self.poly_d[1]),1), dtype=tf.float64)), transform =tfp.bijectors.Sigmoid(to_default_float(-2.), to_default_float(2.)))
        self.g1_poly = gpflow.Parameter(tf.Variable(0.5*tf.random.normal((self.number_of_coefficients(self.poly_d[2]),1), dtype=tf.float64)), transform =tfp.bijectors.Sigmoid(to_default_float(-2.), to_default_float(2.)))
        self.g2_poly = gpflow.Parameter(tf.Variable(0.5*tf.random.normal((self.number_of_coefficients(self.poly_d[3]),1), dtype=tf.float64)), transform =tfp.bijectors.Sigmoid(to_default_float(-2.), to_default_float(2.)))
        self.jitter = jitter_size
        self.invar_neighbourhood = invar_neighbourhood
        self.n_neighbours = n_neighbours

    def number_of_coefficients(self, d):
        c = 0
        for m in range(d+1):
            c+=comb(4+m-1,m)
        return int(c)

    def inv_f1(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1,self.poly_d[0]+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
                for i in index:
                    sub_polynomial_X = tf.multiply(sub_polynomial_X,X[:,i,None])
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.f1_poly))

    def inv_f2(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1, self.poly_d[1]+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
                for i in index:
                    sub_polynomial_X = tf.multiply(sub_polynomial_X,X[:,i,None])
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.f2_poly))

    def inv_g1(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1, self.poly_d[2]+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
                for i in index:
                    sub_polynomial_X = tf.multiply(sub_polynomial_X,X[:,i,None])
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.g1_poly))

    def inv_g2(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1, self.poly_d[3]+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
                for i in index:
                    sub_polynomial_X = tf.multiply(sub_polynomial_X,X[:,i,None])
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.g2_poly))


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

        local_X1_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_X1_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X1_invar_grids.shape), dtype=tf.float64)
        local_X2_invar_grids = tf.repeat(X2, self.n_neighbours, 0) 
        local_X2_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X2_invar_grids.shape), dtype=tf.float64)
        
        local_invar_grids = tf.concat([local_X1_invar_grids, local_X2_invar_grids],0)
        
        Ka1_X1Xg  = self.Ka1(X, local_invar_grids) 
        Ka2_X1Xg  = self.Ka2(X, local_invar_grids) 
        Kv1_X1Xg  = self.Kv1(X, local_invar_grids) 
        Kv2_X1Xg  = self.Kv2(X, local_invar_grids) 
        K_X1Xg = tf.concat([Ka1_X1Xg, Ka2_X1Xg, Kv1_X1Xg, Kv2_X1Xg],0)

        Ka1_X2Xg =  self.Ka1(X2, local_invar_grids) 
        Ka2_X2Xg =  self.Ka2(X2, local_invar_grids) 
        Kv1_X2Xg =  self.Kv1(X2, local_invar_grids)
        Kv2_X2Xg =  self.Kv2(X2, local_invar_grids)
        K_X2Xg = tf.concat([Ka1_X2Xg, Ka2_X2Xg, Kv1_X2Xg, Kv2_X2Xg],0)

        Ka1_XgXg = self.Ka1(local_invar_grids) 
        Ka2_XgXg = self.Ka2(local_invar_grids) 
        Kv1_XgXg = self.Kv1(local_invar_grids) 
        Kv2_XgXg = self.Kv2(local_invar_grids) 
        
        x1_g_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_g1(local_invar_grids)
        x2_g_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_g2(local_invar_grids)
        x1_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_f1(local_invar_grids)
        x2_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_f2(local_invar_grids)
        x_g_1_stacked = tf.concat([x1_g_dot_1, x2_g_dot_1, x1_g_1, x2_g_1],0)

        x1_g_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_g1(local_invar_grids)
        x2_g_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_g2(local_invar_grids)
        x1_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_f1(local_invar_grids)
        x2_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_f2(local_invar_grids)
        x_g_2_stacked = tf.concat([x1_g_dot_2, x2_g_dot_2, x1_g_2, x2_g_2],0)

        x1_g_not_squared = self.inv_g1(local_invar_grids)[:,None]
        x2_g_not_squared = self.inv_g2(local_invar_grids)[:,None]
        x1_g_dot_not_squared = self.inv_f1(local_invar_grids)[:,None]
        x2_g_dot_not_squared = self.inv_f2(local_invar_grids)[:,None]
        x1_g_squared = tf.tensordot(x1_g_not_squared, tf.transpose(x1_g_not_squared), 1)
        x2_g_squared = tf.tensordot(x2_g_not_squared, tf.transpose(x2_g_not_squared), 1)
        x1_g_dot_squared = tf.tensordot(x1_g_dot_not_squared, tf.transpose(x1_g_dot_not_squared), 1)
        x2_g_dot_squared = tf.tensordot(x2_g_dot_not_squared, tf.transpose(x2_g_dot_not_squared), 1)
        
        A = tf.concat([tf.concat([K_X1X1, K_X1X2],1),tf.concat([K_X2X1, K_X2X2],1)],0) 
        B1 = tf.multiply(K_X1Xg, x_g_1_stacked)
        B2 = tf.multiply(K_X2Xg, x_g_2_stacked)
        B = tf.concat([B1, B2], 0)
        C = tf.transpose(B)
        D = tf.multiply(x1_g_dot_squared, Ka1_XgXg) + tf.multiply(x2_g_dot_squared, Ka2_XgXg) + tf.multiply(x1_g_squared, Kv1_XgXg) + tf.multiply(x2_g_squared, Kv2_XgXg) 
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

        local_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_invar_grids += self.invar_neighbourhood*tf.random.normal((local_invar_grids.shape), dtype=tf.float64)
        
        Ka1_Xg  = self.Ka1(X, local_invar_grids) 
        Ka2_Xg  = self.Ka2(X, local_invar_grids) 
        Kv1_Xg  = self.Kv1(X, local_invar_grids) 
        Kv2_Xg  = self.Kv2(X, local_invar_grids) 
        K_Xg = tf.concat([Ka1_Xg, Ka2_Xg, Kv1_Xg, Kv2_Xg],0)

        Ka1_XgXg = self.Ka1(local_invar_grids) 
        Ka2_XgXg = self.Ka2(local_invar_grids) 
        Kv1_XgXg = self.Kv1(local_invar_grids) 
        Kv2_XgXg = self.Kv2(local_invar_grids) 
        
        x1_g = tf.ones([n, 1], dtype=tf.float64) * self.inv_g1(local_invar_grids)
        x2_g = tf.ones([n, 1], dtype=tf.float64) * self.inv_g2(local_invar_grids)
        x1_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.inv_f1(local_invar_grids)
        x2_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.inv_f2(local_invar_grids)
        x_g_stacked = tf.concat([x1_g_dot, x2_g_dot, x1_g, x2_g],0)

        x1_g_not_squared = self.inv_g1(local_invar_grids)[:,None]
        x2_g_not_squared = self.inv_g2(local_invar_grids)[:,None]
        x1_g_dot_not_squared = self.inv_f1(local_invar_grids)[:,None]
        x2_g_dot_not_squared = self.inv_f2(local_invar_grids)[:,None]
        x1_g_squared = tf.tensordot(x1_g_not_squared, tf.transpose(x1_g_not_squared), 1)
        x2_g_squared = tf.tensordot(x2_g_not_squared, tf.transpose(x2_g_not_squared), 1)
        x1_g_dot_squared = tf.tensordot(x1_g_dot_not_squared, tf.transpose(x1_g_dot_not_squared), 1)
        x2_g_dot_squared = tf.tensordot(x2_g_dot_not_squared, tf.transpose(x2_g_dot_not_squared), 1)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(x1_g_dot_squared, Ka1_XgXg) + tf.multiply(x2_g_dot_squared, Ka2_XgXg) + tf.multiply(x1_g_squared, Kv1_XgXg) + tf.multiply(x2_g_squared, Kv2_XgXg) 
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

class Damped_Polynomial_Local_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood, n_neighbours, jitter_size, poly_f_d, poly_g_d):
        super().__init__(active_dims=[0, 1])
        self.poly_f_d = poly_f_d
        self.poly_g_d = poly_g_d
        self.f_poly = gpflow.Parameter(tf.Variable(0.5*tf.random.normal((self.poly_f_d, 1), dtype=tf.float64)), transform =tfp.bijectors.Sigmoid(to_default_float(-2.), to_default_float(2.)))
        self.g_poly = gpflow.Parameter(tf.Variable(0.5*tf.random.normal((self.poly_g_d, 1), dtype=tf.float64)), transform =tfp.bijectors.Sigmoid(to_default_float(-2.), to_default_float(2.)))

        self.n_neighbours = n_neighbours
        self.invar_neighbourhood = invar_neighbourhood
        self.jitter =jitter_size
        self.epsilon = gpflow.Parameter(0.01, transform =tfp.bijectors.Sigmoid(to_default_float(self.jitter), to_default_float(1.)))
        self.Ka = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Kv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 

    def inv_f(self, X):
        return tf.linalg.matmul(tf.math.pow(X, list(range(self.poly_f_d))), self.f_poly)
    def inv_g(self, X):
        return tf.linalg.matmul(tf.math.pow(X, list(range(self.poly_g_d))), self.g_poly)

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

        local_X1_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_X1_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X1_invar_grids.shape), dtype=tf.float64)
        local_X2_invar_grids = tf.repeat(X2, self.n_neighbours, 0) 
        local_X2_invar_grids += self.invar_neighbourhood*tf.random.normal((local_X2_invar_grids.shape), dtype=tf.float64)
        
        local_invar_grids = tf.concat([local_X1_invar_grids, local_X2_invar_grids],0)

        Ka_X1Xg  = self.Ka(X, local_invar_grids) 
        Kv_X1Xg  = self.Kv(X, local_invar_grids) 
        K_X1Xg = tf.concat([Ka_X1Xg, Kv_X1Xg],0)

        Ka_X2Xg =  self.Ka(X2, local_invar_grids) 
        Kv_X2Xg =  self.Kv(X2, local_invar_grids)
        K_X2Xg = tf.concat([Ka_X2Xg, Kv_X2Xg],0)

        Ka_XgXg = self.Ka(local_invar_grids) 
        Kv_XgXg = self.Kv(local_invar_grids) 
        
        x_g_1 = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(local_invar_grids[:,0, None]))
        x_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(local_invar_grids[:,1, None]))
        x_g_1_stacked = tf.concat([x_g_dot_1, x_g_1],0) 
        
        x_g_2 = tf.ones([m, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(local_invar_grids[:,0, None]))
        x_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(local_invar_grids[:,1, None]))
        x_g_2_stacked = tf.concat([x_g_dot_2, x_g_2],0) 

        x_g_squared = tf.tensordot(self.inv_g(local_invar_grids[:,0,None]),tf.transpose(self.inv_g(local_invar_grids[:,0, None])),1)
        x_g_dot_squared = tf.tensordot(self.inv_f(local_invar_grids[:,1,None]),tf.transpose(self.inv_f(local_invar_grids[:,1, None])),1)
        
        A = tf.concat([tf.concat([K_X1X1, K_X1X2],1),tf.concat([K_X2X1, K_X2X2],1)],0) 
        B1 = tf.multiply(K_X1Xg, x_g_1_stacked)
        B2 = tf.multiply(K_X2Xg, x_g_2_stacked)
        B = tf.concat([B1, B2], 0)
        C = tf.transpose(B)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg) 
        D += self.epsilon*tf.eye(D.shape[0], dtype=tf.float64)
        
        return (A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D), 1), C, 1))[:2*n, 2*n:]

    def K_diag(self, X):
        n = X.shape[0]
        zeros_nn =tf.zeros((n,n), dtype=tf.float64)
        
        Ka_X  = self.Ka(X) 
        Kv_X  = self.Kv(X) 
        K_X   = tf.concat([tf.concat([Ka_X,zeros_nn],1),tf.concat([zeros_nn,Kv_X],1)],0)

        local_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_invar_grids += self.invar_neighbourhood*tf.random.normal((local_invar_grids.shape), dtype=tf.float64)

        Ka_Xg  = self.Ka(X, local_invar_grids) 
        Kv_Xg  = self.Kv(X, local_invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(local_invar_grids) 
        Kv_XgXg = self.Kv(local_invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(local_invar_grids[:,0, None]))
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(local_invar_grids[:,1, None]))
        x_g_stacked = tf.concat([x_g_dot, x_g],0)
        
        x_g_squared = tf.tensordot(self.inv_g(local_invar_grids[:,0,None]),tf.transpose(self.inv_g(local_invar_grids[:,0, None])),1)
        x_g_dot_squared = tf.tensordot(self.inv_f(local_invar_grids[:,1,None]),tf.transpose(self.inv_f(local_invar_grids[:,1, None])),1)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg)
        D += self.epsilon*tf.eye(D.shape[0], dtype=tf.float64)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

def get_SHM_Local_Invariance(invar_neighbourhood, n_neighbours, jitter_size):
    invariance_kernel = SHM_Local_Invariance(invar_neighbourhood, n_neighbours,jitter_size)
    invariance_kernel.Ka.variance = gpflow.Parameter(invariance_kernel.Ka.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.variance = gpflow.Parameter(invariance_kernel.Kv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka.lengthscales = gpflow.Parameter(invariance_kernel.Ka.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.lengthscales = gpflow.Parameter(invariance_kernel.Kv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    return invariance_kernel

def get_SHM2D_Local_Invariance(invar_neighbourhood, n_neighbours, jitter_size):
    invariance_kernel = SHM2D_Local_Invariance(invar_neighbourhood, n_neighbours,jitter_size)
    invariance_kernel.Ka1.variance = gpflow.Parameter(invariance_kernel.Ka1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka2.variance = gpflow.Parameter(invariance_kernel.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv1.variance = gpflow.Parameter(invariance_kernel.Kv1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv2.variance = gpflow.Parameter(invariance_kernel.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka1.lengthscales = gpflow.Parameter(invariance_kernel.Ka1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka2.lengthscales = gpflow.Parameter(invariance_kernel.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv1.lengthscales = gpflow.Parameter(invariance_kernel.Kv1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv2.lengthscales = gpflow.Parameter(invariance_kernel.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    return invariance_kernel


def get_Pendulum_Local_Invariance(invar_neighbourhood, n_neighbours, jitter_size):
    invariance_kernel = Pendulum_Local_Invariance(invar_neighbourhood, n_neighbours,  jitter_size)
    invariance_kernel.Ka.variance = gpflow.Parameter(invariance_kernel.Ka.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.variance = gpflow.Parameter(invariance_kernel.Kv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka.lengthscales = gpflow.Parameter(invariance_kernel.Ka.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.lengthscales = gpflow.Parameter(invariance_kernel.Kv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    return invariance_kernel

def get_Double_Pendulum_Local_Invariance(invar_neighbourhood, n_neighbours, jitter_size):
    invariance_kernel = Double_Pendulum_Local_Invariance(invar_neighbourhood, n_neighbours,jitter_size)
    invariance_kernel.Ka1.variance = gpflow.Parameter(invariance_kernel.Ka1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka2.variance = gpflow.Parameter(invariance_kernel.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv1.variance = gpflow.Parameter(invariance_kernel.Kv1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv2.variance = gpflow.Parameter(invariance_kernel.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka1.lengthscales = gpflow.Parameter(invariance_kernel.Ka1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka2.lengthscales = gpflow.Parameter(invariance_kernel.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv1.lengthscales = gpflow.Parameter(invariance_kernel.Kv1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv2.lengthscales = gpflow.Parameter(invariance_kernel.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    return invariance_kernel


def get_Polynomial_Local_Invariance(invar_neighbourhood, n_neighbours, jitter_size, poly_f_d, poly_g_d):
    invariance_kernel = Polynomial_Local_Invariance(invar_neighbourhood, n_neighbours, jitter_size, poly_f_d, poly_g_d)
    invariance_kernel.Ka.variance = gpflow.Parameter(invariance_kernel.Ka.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.variance = gpflow.Parameter(invariance_kernel.Kv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka.lengthscales = gpflow.Parameter(invariance_kernel.Ka.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.lengthscales = gpflow.Parameter(invariance_kernel.Kv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    return invariance_kernel

def get_Polynomial_2D_Local_Invariance(invar_neighbourhood, n_neighbours, jitter_size, poly_d):
    invariance_kernel = Polynomial_2D_Local_Invariance(invar_neighbourhood, n_neighbours,jitter_size, poly_d)
    invariance_kernel.Ka1.variance = gpflow.Parameter(invariance_kernel.Ka1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka2.variance = gpflow.Parameter(invariance_kernel.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv1.variance = gpflow.Parameter(invariance_kernel.Kv1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv2.variance = gpflow.Parameter(invariance_kernel.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka1.lengthscales = gpflow.Parameter(invariance_kernel.Ka1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka2.lengthscales = gpflow.Parameter(invariance_kernel.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv1.lengthscales = gpflow.Parameter(invariance_kernel.Kv1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv2.lengthscales = gpflow.Parameter(invariance_kernel.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    return invariance_kernel

def get_Damped_Polynomial_Local_Invariance(invar_neighbourhood, n_neighbours, jitter_size, poly_f_d, poly_g_d):
    invariance_kernel = Damped_Polynomial_Local_Invariance(invar_neighbourhood, n_neighbours, jitter_size, poly_f_d, poly_g_d)
    invariance_kernel.Ka.variance = gpflow.Parameter(invariance_kernel.Ka.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.variance = gpflow.Parameter(invariance_kernel.Kv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Ka.lengthscales = gpflow.Parameter(invariance_kernel.Ka.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    invariance_kernel.Kv.lengthscales = gpflow.Parameter(invariance_kernel.Kv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
    return invariance_kernel