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

# %%
dt = 0.01
t = np.linspace(0, 30, int(30/dt))
g = 1
l1 = 1
l2 = 1
m1 = 1
m2 = 1
def f(t, r):
    theta1 = r[0]
    theta2 = r[1]
    omega1 = r[2]
    omega2 = r[3]
    return np.array([omega1, omega2, (-g*(2*m1+m2)*np.sin(theta1)-m2*g*np.sin(theta1-2*theta2)-2*np.sin(theta1-theta2)*m2*(omega2**2*l2+omega1**2*l1*np.cos(theta1-theta2)))/(l1*(2*m1+m2-m2*np.cos(2*theta1-2*theta2))), (2*np.sin(theta1-theta2)*(omega1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(theta1)+omega2**2*l2*m2*np.cos(theta1-theta2)))/(l2*(2*m1+m2-m2*np.cos(2*theta1-2*theta2)))])
results = odeint(f, [np.radians(30),np.radians(65), 0, 0], t, tfirst=True)
x1 = results[0::100,0]
x2 = results[0::100,1]
v1 = results[0::100,2]
v2 = results[0::100,3]
t = t[0::100]

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
test_density = 40
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
moi = MOI4()

# %%
m_normal = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,2,None], Y[:,3,None],Y[:,0, None], Y[:,1,None]],1)),(Y.shape[0]*4,1))), kernel=moi, mean_function=Zero_mean(output_dim=2))
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_normal.training_loss, m_normal.trainable_variables, options=dict(maxiter=100))
pred, var = m_normal.predict_f(test_points)
print(m_normal.log_marginal_likelihood().numpy())
print_summary(m_normal)