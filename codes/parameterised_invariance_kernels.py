import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from scipy import interpolate
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from itertools import combinations_with_replacement
from scipy.special import comb

class PolynomialInvariance(gpflow.kernels.Kernel):
    def __init__(self, invariance_range, invar_density, jitter_size, poly_f_d, poly_g_d):
        super().__init__(active_dims=[0,1])
        self.poly_f_d = poly_f_d
        self.poly_g_d = poly_g_d
        self.prior_variance = 0.1#gpflow.Parameter(0.1, transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(1.)))
        self.f_poly = gpflow.Parameter(tf.Variable(1e-3*np.random.normal(size=(self.poly_f_d,1)), dtype=tf.float64), transform =tfp.bijectors.Sigmoid(to_default_float(-1.), to_default_float(1.)), prior=tfp.distributions.Laplace(to_default_float(0),to_default_float(self.prior_variance)), name="f_poly")
        self.g_poly = gpflow.Parameter(tf.Variable(1e-3*np.random.normal(size=(self.poly_g_d,1)), dtype=tf.float64), transform =tfp.bijectors.Sigmoid(to_default_float(-1.), to_default_float(1.)), prior=tfp.distributions.Laplace(to_default_float(0),to_default_float(self.prior_variance)), name="g_poly")

        self.jitter =jitter_size
        self.Ka = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Kv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        invariance_xs = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_vs = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_xx, invariance_vv = tf.meshgrid(invariance_xs, invariance_vs)
        self.invar_grids = tf.stack([tf.reshape(invariance_xx,[-1]), tf.reshape(invariance_vv,[-1])], axis=1)

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
        
        Ka_X1Xg  = self.Ka(X, self.invar_grids) 
        Kv_X1Xg  = self.Kv(X, self.invar_grids) 
        K_X1Xg = tf.concat([Ka_X1Xg, Kv_X1Xg],0)

        Ka_X2Xg =  self.Ka(X2, self.invar_grids) 
        Kv_X2Xg =  self.Kv(X2, self.invar_grids)
        K_X2Xg = tf.concat([Ka_X2Xg, Kv_X2Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g_1 = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(self.invar_grids[:,0, None]))
        x_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(self.invar_grids[:,1, None]))
        x_g_1_stacked = tf.concat([x_g_dot_1, x_g_1],0) 
        
        x_g_2 = tf.ones([m, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(self.invar_grids[:,0, None]))
        x_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(self.invar_grids[:,1, None]))
        x_g_2_stacked = tf.concat([x_g_dot_2, x_g_2],0) 

        x_g_squared = tf.tensordot(self.inv_g(self.invar_grids[:,0,None]),tf.transpose(self.inv_g(self.invar_grids[:,0, None])),1)
        x_g_dot_squared = tf.tensordot(self.inv_f(self.invar_grids[:,1,None]),tf.transpose(self.inv_f(self.invar_grids[:,1, None])),1)
        
        
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
        
        Ka_Xg  = self.Ka(X, self.invar_grids) 
        Kv_Xg  = self.Kv(X, self.invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(self.invar_grids[:,0, None]))
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(self.invar_grids[:,1, None]))
        x_g_stacked = tf.concat([x_g_dot, x_g],0)
        
        x_g_squared = tf.tensordot(self.inv_g(self.invar_grids[:,0,None]),tf.transpose(self.inv_g(self.invar_grids[:,0, None])),1)
        x_g_dot_squared = tf.tensordot(self.inv_f(self.invar_grids[:,1,None]),tf.transpose(self.inv_f(self.invar_grids[:,1, None])),1)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg)
        D += self.jitter*tf.eye(D.shape[0], dtype=tf.float64)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

class DampedPolynomialInvariance(gpflow.kernels.Kernel):
    def __init__(self, invariance_range, invar_density, jitter_size, poly_f_d, poly_g_d):
        super().__init__(active_dims=[0, 1])
        self.poly_f_d = poly_f_d
        self.poly_g_d = poly_g_d
        self.prior_variance = 0.1
        self.f_poly = gpflow.Parameter(tf.Variable(1e-3*np.random.normal(size=(self.poly_f_d,1)), dtype=tf.float64), transform =tfp.bijectors.Sigmoid(to_default_float(-1.), to_default_float(1.)), prior=tfp.distributions.Laplace(to_default_float(0),to_default_float(self.prior_variance)), name="f_poly")
        self.g_poly = gpflow.Parameter(tf.Variable(1e-3*np.random.normal(size=(self.poly_g_d,1)), dtype=tf.float64), transform =tfp.bijectors.Sigmoid(to_default_float(-1.), to_default_float(1.)), prior=tfp.distributions.Laplace(to_default_float(0),to_default_float(self.prior_variance)), name="g_poly")

        self.jitter =jitter_size
        self.epsilon = gpflow.Parameter(0.01, transform =tfp.bijectors.Sigmoid(to_default_float(self.jitter), to_default_float(5.)))
        self.Ka = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Kv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        invariance_xs = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_vs = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_xx, invariance_vv = tf.meshgrid(invariance_xs, invariance_vs)
        self.invar_grids = tf.stack([tf.reshape(invariance_xx,[-1]), tf.reshape(invariance_vv,[-1])], axis=1)

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
        
        Ka_X1Xg  = self.Ka(X, self.invar_grids) 
        Kv_X1Xg  = self.Kv(X, self.invar_grids) 
        K_X1Xg = tf.concat([Ka_X1Xg, Kv_X1Xg],0)

        Ka_X2Xg =  self.Ka(X2, self.invar_grids) 
        Kv_X2Xg =  self.Kv(X2, self.invar_grids)
        K_X2Xg = tf.concat([Ka_X2Xg, Kv_X2Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g_1 = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(self.invar_grids[:,0, None]))
        x_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(self.invar_grids[:,1, None]))
        x_g_1_stacked = tf.concat([x_g_dot_1, x_g_1],0) 
        
        x_g_2 = tf.ones([m, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(self.invar_grids[:,0, None]))
        x_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(self.invar_grids[:,1, None]))
        x_g_2_stacked = tf.concat([x_g_dot_2, x_g_2],0) 

        x_g_squared = tf.tensordot(self.inv_g(self.invar_grids[:,0,None]),tf.transpose(self.inv_g(self.invar_grids[:,0, None])),1)
        x_g_dot_squared = tf.tensordot(self.inv_f(self.invar_grids[:,1,None]),tf.transpose(self.inv_f(self.invar_grids[:,1, None])),1)
        
        
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
        
        Ka_Xg  = self.Ka(X, self.invar_grids) 
        Kv_Xg  = self.Kv(X, self.invar_grids) 
        K_Xg = tf.concat([Ka_Xg, Kv_Xg],0)

        Ka_XgXg = self.Ka(self.invar_grids) 
        Kv_XgXg = self.Kv(self.invar_grids) 
        
        x_g = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_g(self.invar_grids[:,0, None]))
        x_g_dot = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze(self.inv_f(self.invar_grids[:,1, None]))
        x_g_stacked = tf.concat([x_g_dot, x_g],0)
        
        x_g_squared = tf.tensordot(self.inv_g(self.invar_grids[:,0,None]),tf.transpose(self.inv_g(self.invar_grids[:,0, None])),1)
        x_g_dot_squared = tf.tensordot(self.inv_f(self.invar_grids[:,1,None]),tf.transpose(self.inv_f(self.invar_grids[:,1, None])),1)
        
        A = K_X
        B = tf.multiply(K_Xg, x_g_stacked)
        C = tf.transpose(B)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg)
        D += self.epsilon*tf.eye(D.shape[0], dtype=tf.float64)
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

class PolynomialInvariance2D(gpflow.kernels.Kernel):
    def __init__(self,  invariance_range, invar_density, jitter_size, poly_d):
        super().__init__(active_dims=[0, 1, 2, 3])
        #poly_d = [d, d, d, d]
        self.Ka1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Ka2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv1 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.Kv2 = gpflow.kernels.RBF(variance=1, lengthscales=[1,1,1,1])
        self.poly_d = poly_d

        init_poly = np.zeros((4, self.number_of_coefficients(self.poly_d)))
        init_poly[0,3]=1
        init_poly[1,4]=1
        init_poly[2,1]=1
        init_poly[3,2]=1
        '''
        init_poly[0,3]=2;init_poly[0,4]=1;init_poly[0,18]=-0.5;init_poly[0,27]=-0.5;init_poly[0,21]=1
        init_poly[1,3]=1;init_poly[1,4]=1;init_poly[0,17]=-0.5;init_poly[0,26]=-0.5;init_poly[0,20]=1
        init_poly[2,1]=2;init_poly[2,23]=-1;init_poly[0,29]=1
        init_poly[3,2]=1;init_poly[3,23]=1;init_poly[0,29]=-1
        '''
        init_poly = tf.Variable(init_poly/2, dtype=tf.float64)
        self.poly = gpflow.Parameter(init_poly, transform =tfp.bijectors.Sigmoid(to_default_float(-1.), to_default_float(1.)), trainable=True, name="poly")#, prior=tfp.distributions.Laplace(to_default_float(0),(self.prior_variance)), name="poly")

        self.jitter = jitter_size

        invariance_x1s = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_x2s = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_v1s = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_v2s = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_xx1, invariance_xx2, invariance_vv1, invariance_vv2 = tf.meshgrid(invariance_x1s, invariance_x2s, invariance_v1s, invariance_v2s)
        self.invar_grids = tf.stack([tf.reshape(invariance_xx1,[-1]), tf.reshape(invariance_xx2,[-1]), tf.reshape(invariance_vv1,[-1]), tf.reshape(invariance_vv2,[-1])], axis=1)

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
                for i in index:
                   sub_polynomial_X = tf.multiply(sub_polynomial_X,X[:,i,None])
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.poly[0,:, None]))

    def inv_f2(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1, self.poly_d+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
                for i in index:
                   sub_polynomial_X = tf.multiply(sub_polynomial_X,X[:,i,None])
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.poly[1,:, None]))

    def inv_g1(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1, self.poly_d+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
                for i in index:
                   sub_polynomial_X = tf.multiply(sub_polynomial_X,X[:,i,None])
                polynomial_X = tf.concat([polynomial_X, sub_polynomial_X], 1)

        return tf.squeeze(tf.linalg.matmul(polynomial_X, self.poly[2,:, None]))

    def inv_g2(self, X):
        polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
        for d in range(1, self.poly_d+1):
            indices = list(combinations_with_replacement(range(4),d))
            for index in indices:
                sub_polynomial_X = tf.ones((X.shape[0],1),dtype=tf.float64)
#                for i in [0, 1, 2, 3]:
#                    sub_polynomial_X = tf.multiply(sub_polynomial_X,tf.math.polyval(list(legendre(index.count(i))),X[:,i,None]))
                for i in index:
                   sub_polynomial_X = tf.multiply(sub_polynomial_X,X[:,i,None])
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
        
        x1_g_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_g1(self.invar_grids)
        x2_g_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_g2(self.invar_grids)
        x1_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_f1(self.invar_grids)
        x2_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * self.inv_f2(self.invar_grids)
        x_g_1_stacked = tf.concat([x1_g_dot_1, x2_g_dot_1, x1_g_1, x2_g_1],0)

        x1_g_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_g1(self.invar_grids)
        x2_g_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_g2(self.invar_grids)
        x1_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_f1(self.invar_grids)
        x2_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) * self.inv_f2(self.invar_grids)
        x_g_2_stacked = tf.concat([x1_g_dot_2, x2_g_dot_2, x1_g_2, x2_g_2],0)

        x1_g_not_squared = self.inv_g1(self.invar_grids)[:,None]
        x2_g_not_squared = self.inv_g2(self.invar_grids)[:,None]
        x1_g_dot_not_squared = self.inv_f1(self.invar_grids)[:,None]
        x2_g_dot_not_squared = self.inv_f2(self.invar_grids)[:,None]
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

        Ka1_Xg  = self.Ka1(X, self.invar_grids) 
        Ka2_Xg  = self.Ka2(X, self.invar_grids) 
        Kv1_Xg  = self.Kv1(X, self.invar_grids) 
        Kv2_Xg  = self.Kv2(X, self.invar_grids) 
        K_Xg = tf.concat([Ka1_Xg, Ka2_Xg, Kv1_Xg, Kv2_Xg],0)

        Ka1_XgXg = self.Ka1(self.invar_grids) 
        Ka2_XgXg = self.Ka2(self.invar_grids) 
        Kv1_XgXg = self.Kv1(self.invar_grids) 
        Kv2_XgXg = self.Kv2(self.invar_grids) 
        
        x1_g = tf.ones([n, 1], dtype=tf.float64) * self.inv_g1(self.invar_grids)
        x2_g = tf.ones([n, 1], dtype=tf.float64) * self.inv_g2(self.invar_grids)
        x1_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.inv_f1(self.invar_grids)
        x2_g_dot = tf.ones([n, 1], dtype=tf.float64) * self.inv_f2(self.invar_grids)
        x_g_stacked = tf.concat([x1_g_dot, x2_g_dot, x1_g, x2_g],0)

        x1_g_not_squared = self.inv_g1(self.invar_grids)[:,None]
        x2_g_not_squared = self.inv_g2(self.invar_grids)[:,None]
        x1_g_dot_not_squared = self.inv_f1(self.invar_grids)[:,None]
        x2_g_dot_not_squared = self.inv_f2(self.invar_grids)[:,None]
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

def get_polynomial_invariance(invar_range, invar_density, jitter_size, poly_f_d, poly_g_d):
    invariance_kernel = PolynomialInvariance(invar_range, invar_density, jitter_size, poly_f_d, poly_g_d)
    invariance_kernel.Ka.variance = gpflow.Parameter(invariance_kernel.Ka.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(10.))) 
    invariance_kernel.Kv.variance = gpflow.Parameter(invariance_kernel.Kv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(10.))) 
    invariance_kernel.Ka.lengthscales = gpflow.Parameter(invariance_kernel.Ka.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(10.))) 
    invariance_kernel.Kv.lengthscales = gpflow.Parameter(invariance_kernel.Kv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(10.))) 
    return invariance_kernel

def get_damped_polynomial_invariance(invar_range, invar_density, jitter_size, poly_f_d, poly_g_d):
    invariance_kernel = DampedPolynomialInvariance(invar_range, invar_density, jitter_size, poly_f_d, poly_g_d)
    invariance_kernel.Ka.variance = gpflow.Parameter(invariance_kernel.Ka.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(10.))) 
    invariance_kernel.Kv.variance = gpflow.Parameter(invariance_kernel.Kv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(10.))) 
    invariance_kernel.Ka.lengthscales = gpflow.Parameter(invariance_kernel.Ka.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(10.))) 
    invariance_kernel.Kv.lengthscales = gpflow.Parameter(invariance_kernel.Kv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(10.))) 
    return invariance_kernel

def get_polynomial_invariance_2D(invar_range, invar_density, jitter_size, poly_d):
    invariance_kernel = PolynomialInvariance2D(invar_range, invar_density, jitter_size, poly_d)
    invariance_kernel.Ka1.variance = gpflow.Parameter(invariance_kernel.Ka1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka2.variance = gpflow.Parameter(invariance_kernel.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv1.variance = gpflow.Parameter(invariance_kernel.Kv1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv2.variance = gpflow.Parameter(invariance_kernel.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka1.lengthscales = gpflow.Parameter(invariance_kernel.Ka1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Ka2.lengthscales = gpflow.Parameter(invariance_kernel.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv1.lengthscales = gpflow.Parameter(invariance_kernel.Kv1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    invariance_kernel.Kv2.lengthscales = gpflow.Parameter(invariance_kernel.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
    return invariance_kernel 

