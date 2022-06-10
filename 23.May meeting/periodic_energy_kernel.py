
# %%
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from gpflow.utilities import print_summary, positive
from gpflow.utilities.ops import broadcasting_elementwise


# %%
# an example of simple harmonic motion
t = tf.linspace(0, 100, 100)
x = tf.math.sin(t)
v = tf.math.cos(t)
plt.plot(t, x, "--")
plt.plot(t, v, "--")
#sampled_t = list(sorted(random.sample(list(t), 100)))
#sampled_x = 5*np.cos(sampled_t)
#sampled_v = 5*np.sin(sampled_t)
#plt.plot(sampled_t, sampled_x, 'x')
#plt.plot(sampled_t, sampled_v, 'x')
# %%
#plt.plot(x,v)
#plt.plot(sampled_x, sampled_v, "x")

# %%
X = tf.concat([x[:,None], v[:,None], t[:,None]], axis=-1)
Y = (X[2:,:]-X[:-2, :])/2/(100/100)
X = X[1:-1, :]
plt.plot(X[:,1])
plt.plot(Y[:,0])

# %%
class Energy(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(active_dims=[0,1])
        self.variance = gpflow.Parameter(1.0, transform=positive())
        self.lengthscale = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        Xs = tf.reduce_sum(tf.square(X), -1)
        X2s = tf.reduce_sum(tf.square(X2), -1)
        return self.variance * tf.exp(-broadcasting_elementwise(tf.subtract, Xs, X2s)/(2*self.lengthscale))

    def K_diag(self, X):
        return self.variance * tf.ones((X.shape[0]),dtype=tf.float64) # this returns a 1D tensor


energy = Energy()
# %%
energy_time = Energy()*gpflow.kernels.Cosine(active_dims=[2])

# %%
m = gpflow.models.GPR(data=(X, Y[:, 1,None]), kernel=energy_time, mean_function=None)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=500))
print(m.log_marginal_likelihood())


# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.scatter(X[:,0], X[:,1], X[:,2],c = Y[:,1,None], cmap="viridis", marker="o", s=3) #data
ax.view_init(10,-65)
ax.set_xlabel("position")
ax.set_ylabel("velocity")
ax.set_zlabel("time")

# %%
test_range =  1
test_density = 5
time_range = 100
test_xs = tf.linspace(-test_range,test_range,test_density)
test_vs = tf.linspace(-test_range,test_range,test_density)
test_ts = tf.linspace(-time_range,time_range,int(time_range/5))
test_xx, test_vv, test_tt = tf.meshgrid(test_xs, test_vs, test_ts)
test_points = tf.stack([tf.reshape(test_xx,[-1]), tf.reshape(test_vv,[-1]), tf.reshape(test_tt,[-1])], axis=1)

pred, var = m.predict_f(tf.cast(test_points, tf.float64))

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(test_xx, test_vv, tf.reshape(pred, test_xx.shape)-1.96*tf.math.sqrt(tf.reshape(tf.linalg.tensor_diag_part(var), test_xx.shape)), color="grey",linewidth=0, antialiased=False, alpha=0.1)
#surf = ax.plot_surface(test_xx, test_vv, -test_xx, color="grey",linewidth=0, antialiased=False, alpha=0.1)
#surf = ax.scatter(test_xx, test_vv, tf.reshape(pred, test_xx.shape), cmap="viridis",linewidth=0, antialiased=False, alpha=0.3)
surf = ax.scatter(test_xx, test_vv, test_tt,c = tf.reshape(pred, test_xx.shape), cmap="viridis", marker="o", s=1) #data
#inv_p = ax.scatter(invariance_points[:,0], invariance_points[:,1], tf.reshape(pred_inv, invariance_points[:,0].shape), color="red", marker="o", s=3, alpha=0.2)
ax.view_init(10,-65)
#cax = fig.add_axes([0.9, .3, 0.02, 0.6])
#fig.colorbar(surf)
ax.set_xlabel("position")
ax.set_ylabel("velocity")
ax.set_zlabel("time")

# %%
