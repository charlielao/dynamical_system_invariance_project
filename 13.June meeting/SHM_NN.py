######
# first version, contains SHM kernel, SHM parameter learning kernel, standard GP kernel, prior plot, posterior plot, degree of freedom calculation as well as samples from a kernel 
######
# %%
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import random
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from termcolor import colored

# %%
dt = 1
t = tf.linspace(0, 30, int(30/dt))
x = tf.math.sin(t)
v = tf.math.cos(t)
plt.plot(t, x, "--")
plt.plot(t, v, "--")
# to sample the data randomly instead of regular spacing
#sampled_t = list(sorted(random.sample(list(t), 50)))
#sampled_x = tf.math.cos(sampled_t)
#sampled_v = tf.math.sin(sampled_t)
#plt.plot(sampled_t, sampled_x, 'x')
#plt.plot(sampled_t, sampled_v, 'x')
#plt.plot(x,v)
#plt.plot(sampled_x, sampled_v, "x")

# %%
X1 = tf.concat([x[:,None], v[:,None]], axis=-1)
X1 += tf.random.normal((X1.shape), 0, 0.1, dtype=tf.float64)
X2 = 2*X1
X2 += tf.random.normal((X2.shape), 0, 0.1, dtype=tf.float64)
Y1 = (X1[2:,:]-X1[:-2, :])/(2*dt) # to estimate acceleration and velocity by discrete differenation
Y2 = (X2[2:,:]-X2[:-2, :])/(2*dt) # to estimate acceleration and velocity by discrete differenation
X1 = X1[1:-1, :]
X2 = X2[1:-1, :]
X = tf.concat([X1, X2], axis=0)
Y = tf.concat([Y1, Y2], axis=0) 
plt.plot(X[:,1])
plt.plot(Y[:,0])
# %%
# plotting

def plotting(pred, var, eval_points, data, save, name, angle1, angle2, acc, lml):
    X, Y = data
    test_xx, test_vv = eval_points
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(test_xx, test_vv, tf.reshape(pred, test_xx.shape), cmap="viridis",linewidth=0, antialiased=False, alpha=0.5)
    surf = ax.plot_surface(test_xx, test_vv, tf.reshape(pred, test_xx.shape)+1.96*tf.math.sqrt(tf.reshape(var, test_xx.shape)), color="grey",linewidth=0, antialiased=False, alpha=0.1)
    surf = ax.plot_surface(test_xx, test_vv, tf.reshape(pred, test_xx.shape)-1.96*tf.math.sqrt(tf.reshape(var, test_xx.shape)), color="grey",linewidth=0, antialiased=False, alpha=0.1)
    if acc:
        ax.scatter(X[:,0], X[:,1],Y[:,1,None], color="black", marker="o", s=3) #data
    else:
        ax.scatter(X[:,0], X[:,1],Y[:,0,None], color="black", marker="o", s=3) #data
    ax.view_init(angle1,angle2)
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    if acc:
        ax.set_zlabel("acceleration")
    else:
        ax.set_zlabel("velocity")
    ax.annotate("log marginal likelihood: {0:0.2f}".format(lml), xy=(0.5, 0.9), xycoords='axes fraction')
    if save:
        plt.savefig(name+"_3D.pdf")
    plt.figure(figsize=(5, 3))
    contours = plt.contourf(test_xx, test_vv, tf.reshape(pred,(test_xx.shape)), levels=100, cmap="viridis", alpha=0.3)
    plt.colorbar(contours)
    if acc:
        contours = plt.scatter(X[:,0],X[:,1],c=Y[:,1,None],cmap="viridis", alpha=0.2)
    else:
        contours = plt.scatter(X[:,0],X[:,1],c=Y[:,0,None],cmap="viridis", alpha=0.2)
    plt.xlim((-test_range, test_range))
    plt.ylim((-test_range, test_range))
    plt.xlabel("position")
    plt.ylabel("velocity")
    if save:
        plt.savefig(name+"_contour.pdf")
# range we are evaluating the test points on
test_range = 3 
test_density = 40
test_xs = tf.linspace(-test_range,test_range,test_density)
test_vs = tf.linspace(-test_range,test_range,test_density)
test_xx, test_vv = tf.meshgrid(test_xs, test_vs)
test_points = tf.stack([tf.reshape(test_xx,[-1]), tf.reshape(test_vv,[-1])], axis=1)
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
class MOI(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(active_dims=[0,1])
        self.RBFa = gpflow.kernels.RBF(variance=1, lengthscales=[1,1])
        self.RBFv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1])
        self.Ka = self.RBFa
        self.Kv = self.RBFv
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


# %%
moi = MOI()
moi.RBFa.variance = gpflow.Parameter(moi.RBFa.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
moi.RBFv.variance = gpflow.Parameter(moi.RBFv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
moi.RBFa.lengthscales = gpflow.Parameter(moi.RBFa.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
moi.RBFv.lengthscales = gpflow.Parameter(moi.RBFv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(10.))) 
#%%
m_normal = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,1,None],Y[:,0, None]],1)),(Y.shape[0]*2,1))), kernel=moi, mean_function=Zero_mean(output_dim=2))
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_normal.training_loss, m_normal.trainable_variables, options=dict(maxiter=100))
pred, var = m_normal.predict_f(test_points)
print(m_normal.log_marginal_likelihood().numpy())
print_summary(m_normal)
# %%
plotting(pred[:int(pred.shape[0]/2),:], var[:int(var.shape[0]/2),:], eval_points=(test_xx, test_vv), data=(X,Y),save=0, name="", angle1=10, angle2=-65, acc=1, lml=m_normal.log_marginal_likelihood().numpy())
plotting(pred[int(pred.shape[0]/2):,:], var[int(var.shape[0]/2):,:], eval_points=(test_xx, test_vv), data=(X,Y),save=0, name="", angle1=10, angle2=-65, acc=0, lml=m_normal.log_marginal_likelihood().numpy())
# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense ,Input

# %%
class SHO_Energy_Invariance_unknown_parameter(gpflow.kernels.Kernel):
    def __init__(self, invariance_range, invar_density):
        super().__init__(active_dims=[0, 1])

        self.weight_f_0 = gpflow.Parameter(tf.reshape(tf.Variable([1.]*1),(1,1)),transform =tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.)) )
#        self.bias_f_0 = gpflow.Parameter(tf.reshape(1.,(1,1)),transform =tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.)) )
        self.weight_g_0 = gpflow.Parameter(tf.reshape(tf.Variable([1.]*1),(1,1)),transform =tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.)) )
#        self.bias_g_0 = gpflow.Parameter(tf.reshape(1.,(1,1)),transform =tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.)) )
#        self.weight_f_1 = gpflow.Parameter(tf.reshape(tf.Variable([1.]*1),(1,1)),transform =tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.)) )
#        self.bias_f_1 = gpflow.Parameter(tf.reshape(1.,(1,1)),transform =tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.)) )
#        self.weight_g_1 = gpflow.Parameter(tf.reshape(tf.Variable([1.]*1),(1,1)),transform =tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.)) )
#        self.bias_g_1 = gpflow.Parameter(tf.reshape(1.,(1,1)),transform =tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.)) )

        self.jitter = gpflow.kernels.White(1e-5)
        self.RBFa = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.RBFv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1]) 
        self.Ka =  self.RBFa + self.jitter
        self.Kv =  self.RBFv + self.jitter
        invariance_xs = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_vs = tf.linspace(-invariance_range,invariance_range,invar_density)
        invariance_xx, invariance_vv = tf.meshgrid(invariance_xs, invariance_vs)
        self.invar_grids = tf.stack([tf.reshape(invariance_xx,[-1]), tf.reshape(invariance_vv,[-1])], axis=1)

    def inv_f(self, X):
#        return tf.tensordot(tf.math.sigmoid(tf.tensordot(X, self.weight_f_0, 1)+self.bias_f_0), self.weight_f_1,1)+self.bias_f_1
        return tf.tensordot(X, self.weight_f_0, 1)
        
    def inv_g(self, X):
        return tf.tensordot(X, self.weight_g_0, 1)
#        return tf.tensordot(tf.math.sigmoid(tf.tensordot(X, self.weight_g_0, 1)+self.bias_g_0), self.weight_g_1,1)+self.bias_g_1

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
        x_g_dot_1 = tf.ones([n, 1], dtype=tf.float64) * tf.squeeze( self.inv_f(self.invar_grids[:,1, None]))
        x_g_1_stacked = tf.concat([x_g_dot_1, x_g_1],0) 
        
        x_g_2 = tf.ones([m, 1], dtype=tf.float64) *  tf.squeeze(self.inv_g(self.invar_grids[:,0, None]))
        x_g_dot_2 = tf.ones([m, 1], dtype=tf.float64) *  tf.squeeze(self.inv_f(self.invar_grids[:,1, None]))
        x_g_2_stacked = tf.concat([x_g_dot_2, x_g_2],0) 

        x_g_squared = tf.tensordot(self.inv_g(self.invar_grids[:,0,None]),tf.transpose(self.inv_g(self.invar_grids[:,0, None])),1) 
        x_g_dot_squared = tf.tensordot(self.inv_f(self.invar_grids[:,1,None]),tf.transpose(self.inv_f(self.invar_grids[:,1, None])),1)
        
        A = tf.concat([tf.concat([K_X1X1, K_X1X2],1),tf.concat([K_X2X1, K_X2X2],1)],0) 
        B1 = tf.multiply(K_X1Xg, x_g_1_stacked)
        B2 = tf.multiply(K_X2Xg, x_g_2_stacked)
        B = tf.concat([B1, B2], 0)
        C = tf.transpose(B)
        D = tf.multiply(x_g_dot_squared, Ka_XgXg) + tf.multiply(x_g_squared, Kv_XgXg) 
        
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
        
        return tf.linalg.tensor_diag_part(A-tf.tensordot(tf.tensordot(B, tf.linalg.inv(D),1), C, 1))

# %%
energy_kernel_unknown = SHO_Energy_Invariance_unknown_parameter(3, 20)
set_trainable(energy_kernel_unknown.jitter.variance, False)
energy_kernel_unknown.RBFa.variance = gpflow.Parameter(energy_kernel_unknown.RBFa.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
energy_kernel_unknown.RBFv.variance = gpflow.Parameter(energy_kernel_unknown.RBFv.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
energy_kernel_unknown.RBFa.lengthscales = gpflow.Parameter(energy_kernel_unknown.RBFa.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
energy_kernel_unknown.RBFv.lengthscales = gpflow.Parameter(energy_kernel_unknown.RBFv.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(5.))) 
# %%
energy_kernel_unknown(X)
#%%
m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,1,None],Y[:,0,None]],1)),(Y.shape[0]*2,1))), kernel=energy_kernel_unknown, mean_function=Zero_mean(output_dim=2))

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=200))
pred, var = m.predict_f(test_points)
print_summary(m)
print(m.log_marginal_likelihood().numpy())
# %%
plotting(pred[:int(pred.shape[0]/2),:], var[:int(var.shape[0]/2),:], eval_points=(test_xx, test_vv), data=(X,Y),save=0, name="", angle1=10, angle2=-65, acc=1, lml=m.log_marginal_likelihood().numpy())
plotting(pred[int(pred.shape[0]/2):,:], var[int(var.shape[0]/2):,:], eval_points=(test_xx, test_vv), data=(X,Y),save=0, name="", angle1=10, angle2=-65, acc=0, lml=m.log_marginal_likelihood().numpy())

# %%