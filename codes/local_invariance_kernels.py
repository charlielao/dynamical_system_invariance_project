import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from scipy.special import comb, legendre
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from itertools import combinations_with_replacement


class SHMLocalInvariance2D(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood_min, invar_neighbourhood_max, n_neighbours, jitter_size):
        super().__init__(active_dims=[0, 1, 2, 3])
        self.Ka1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Ka2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.jitter = jitter_size
        self.n_neighbours = n_neighbours
        self.local_invar_grid = tf.multiply(tf.random.uniform((n_neighbours,4), invar_neighbourhood_min,invar_neighbourhood_max, dtype=tf.float64),2*tf.cast(tf.reshape(tf.random.categorical(tf.math.log([[0.5, 0.5]]), 4*n_neighbours), (n_neighbours,4)), tf.float64)-1*tf.ones((n_neighbours,4),dtype=tf.float64))
        self.n_neighbours = n_neighbours


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
        local_X1_invar_grids += tf.tile(self.local_invar_grid, [X.shape[0],1])
        local_X2_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_X2_invar_grids += tf.tile(self.local_invar_grid, [X.shape[0],1])

        local_invar_grids = tf.concat([local_X1_invar_grids, local_X2_invar_grids],0)

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
        local_invar_grids += tf.tile(self.local_invar_grid, [X.shape[0],1])

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

class DoublePendulumLocalInvariance(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood_min, invar_neighbourhood_max, n_neighbours, jitter_size):
        super().__init__(active_dims=[0, 1, 2, 3])
        self.Ka1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Ka2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.jitter = jitter_size
        self.invar_neighbourhood_min = invar_neighbourhood_min
        self.invar_neighbourhood_max = invar_neighbourhood_max
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
        local_X1_invar_grids += tf.multiply(tf.random.uniform((local_X1_invar_grids.shape), self.invar_neighbourhood_min, self.invar_neighbourhood_max, dtype=tf.float64),2*tf.cast(tf.reshape(tf.random.categorical(tf.math.log([[0.5, 0.5]]), tf.math.reduce_prod(local_X1_invar_grids.shape)), (local_X1_invar_grids.shape)), tf.float64)-1*tf.ones((local_X1_invar_grids.shape),dtype=tf.float64))
        local_X2_invar_grids = tf.repeat(X2, self.n_neighbours, 0) 
        local_X2_invar_grids += tf.multiply(tf.random.uniform((local_X2_invar_grids.shape), self.invar_neighbourhood_min, self.invar_neighbourhood_max, dtype=tf.float64),2*tf.cast(tf.reshape(tf.random.categorical(tf.math.log([[0.5, 0.5]]), tf.math.reduce_prod(local_X2_invar_grids.shape)), (local_X2_invar_grids.shape)), tf.float64)-1*tf.ones((local_X2_invar_grids.shape),dtype=tf.float64))
        
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
        local_invar_grids += tf.multiply(tf.random.uniform((local_invar_grids.shape), self.invar_neighbourhood_min, self.invar_neighbourhood_max, dtype=tf.float64),2*tf.cast(tf.reshape(tf.random.categorical(tf.math.log([[0.5, 0.5]]), tf.math.reduce_prod(local_invar_grids.shape)), (local_invar_grids.shape)), tf.float64)-1*tf.ones((local_invar_grids.shape),dtype=tf.float64))

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

class PolynomialLocalInvariance2D(gpflow.kernels.Kernel):
    def __init__(self, invar_neighbourhood_min, invar_neighbourhood_max, n_neighbours, jitter_size, poly_d):
        super().__init__(active_dims=[0, 1, 2, 3])
        #poly_d = [d, d, d, d]
        self.Ka1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Ka2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.poly_d = poly_d
        self.prior_variance = 1e-1
        self.poly = gpflow.Parameter(tf.Variable(1*np.random.normal(size=(4, self.number_of_coefficients(self.poly_d))), dtype=tf.float64), transform =tfp.bijectors.Sigmoid(to_default_float(-5.), to_default_float(5.)), trainable=True, prior=tfp.distributions.Laplace(to_default_float(0),to_default_float(self.prior_variance)))
        self.jitter = jitter_size
        self.local_invar_grid = tf.multiply(tf.random.uniform((n_neighbours,4), invar_neighbourhood_min,invar_neighbourhood_max, dtype=tf.float64),2*tf.cast(tf.reshape(tf.random.categorical(tf.math.log([[0.5, 0.5]]), 4*n_neighbours), (n_neighbours,4)), tf.float64)-1*tf.ones((n_neighbours,4),dtype=tf.float64))
        self.n_neighbours = n_neighbours

    def number_of_coefficients(self, d):
        c = 0
        for m in range(d+1):
            c+=comb(4+m-1,m)
        return int(c)

    def inv_f1(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1,self.poly_d+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
                for i in [0, 1, 2, 3]:
                    sub_polynomial_X = tf.multiply(sub_polynomial_X,tf.math.polyval(list(legendre(index.count(i))),X[:,i,None]))
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.poly[0,:, None]))

    def inv_f2(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1, self.poly_d+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
                for i in [0, 1, 2, 3]:
                    sub_polynomial_X = tf.multiply(sub_polynomial_X,tf.math.polyval(list(legendre(index.count(i))),X[:,i,None]))
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.poly[1,:, None]))

    def inv_g1(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1, self.poly_d+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
                for i in [0, 1, 2, 3]:
                    sub_polynomial_X = tf.multiply(sub_polynomial_X,tf.math.polyval(list(legendre(index.count(i))),X[:,i,None]))
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.poly[2,:, None]))

    def inv_g2(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1, self.poly_d+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
                for i in [0, 1, 2, 3]:
                    sub_polynomial_X = tf.multiply(sub_polynomial_X,tf.math.polyval(list(legendre(index.count(i))),X[:,i,None]))
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.poly[3,:, None]))


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
        local_X1_invar_grids += tf.tile(self.local_invar_grid, [X.shape[0],1])
        local_X2_invar_grids = tf.repeat(X, self.n_neighbours, 0) 
        local_X2_invar_grids += tf.tile(self.local_invar_grid, [X.shape[0],1])

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
        local_invar_grids += tf.tile(self.local_invar_grid, [X.shape[0],1])

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

def get_SHM_local_invariance_2D(invar_neighbourhood_min, invar_neighbourhood_max, n_neighbours, jitter_size):
    invariance_kernel = SHMLocalInvariance2D(invar_neighbourhood_min, invar_neighbourhood_max, n_neighbours,jitter_size)
    invariance_kernel.Ka1.variance = gpflow.Parameter(invariance_kernel.Ka1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka2.variance = gpflow.Parameter(invariance_kernel.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv1.variance = gpflow.Parameter(invariance_kernel.Kv1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv2.variance = gpflow.Parameter(invariance_kernel.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka1.lengthscales = gpflow.Parameter(invariance_kernel.Ka1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka2.lengthscales = gpflow.Parameter(invariance_kernel.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv1.lengthscales = gpflow.Parameter(invariance_kernel.Kv1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv2.lengthscales = gpflow.Parameter(invariance_kernel.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    return invariance_kernel

def get_double_pendulum_local_invariance(invar_neighbourhood_min, invar_neighbourhood_max, n_neighbours, jitter_size):
    invariance_kernel = DoublePendulumLocalInvariance(invar_neighbourhood_min, invar_neighbourhood_max, n_neighbours,jitter_size)
    invariance_kernel.Ka1.variance = gpflow.Parameter(invariance_kernel.Ka1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka2.variance = gpflow.Parameter(invariance_kernel.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv1.variance = gpflow.Parameter(invariance_kernel.Kv1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv2.variance = gpflow.Parameter(invariance_kernel.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka1.lengthscales = gpflow.Parameter(invariance_kernel.Ka1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka2.lengthscales = gpflow.Parameter(invariance_kernel.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv1.lengthscales = gpflow.Parameter(invariance_kernel.Kv1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv2.lengthscales = gpflow.Parameter(invariance_kernel.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    return invariance_kernel

def get_polynomial_local_invariance_2D(invar_neighbourhood_min, invar_neighbourhood_max, n_neighbours, jitter_size, poly_d):
    invariance_kernel = PolynomialLocalInvariance2D(invar_neighbourhood_min, invar_neighbourhood_max, n_neighbours,jitter_size, poly_d)
    invariance_kernel.Ka2.variance = 1.*gpflow.Parameter(invariance_kernel.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka2.variance = 1.*gpflow.Parameter(invariance_kernel.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv2.variance = 1.*gpflow.Parameter(invariance_kernel.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv2.variance = 1.*gpflow.Parameter(invariance_kernel.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka2.lengthscales = 1.*gpflow.Parameter(invariance_kernel.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka2.lengthscales = 1.*gpflow.Parameter(invariance_kernel.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv2.lengthscales = 1.*gpflow.Parameter(invariance_kernel.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv2.lengthscales = 1.*gpflow.Parameter(invariance_kernel.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    return invariance_kernel
