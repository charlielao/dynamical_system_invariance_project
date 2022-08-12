# %%
# %%
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from scipy.misc import derivative
from scipy import interpolate
import random
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from termcolor import colored
# %%
X = np.random.uniform(-5, 5, 5)[:,None]
Y =((X+X**2)*np.sin(X))
x = np.linspace(-5, 5, 100)
y = (x+x**2)*np.sin(x)
# %%

def callback(step, variables, values):
    print([x for x in variables if x.name == "softplus:0"])

m = gpflow.models.GPR(data=(X, Y), kernel=gpflow.kernels.RBF(), mean_function=None)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=0), step_callback=callback)

xx = np.linspace(-5, 5, 100).reshape(100, 1)  # test points must be of shape (N, D)
mean, var = m.predict_f(xx)
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
plt.plot(X, Y, "kx", mew=2, label="data")
plt.plot(xx, y, color="orange", label="truth")
plt.plot(xx, mean, "C0", lw=2, label="prediction")
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.1,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-5, 5)
plt.legend()
#plt.savefig("figures/posterior.pdf")



    

# %%
fig, (ax0, ax1, ax2) = plt.subplots(3)
kernel = gpflow.kernels.Matern52()
K = kernel(xx)
f_samples = np.random.multivariate_normal(np.zeros(100), K, size=5)
ax0.plot(xx, f_samples.T, "C0", linewidth=0.5)
ax0.fill_between(
    xx[:,0],
    -1.96 * np.sqrt(np.diag(K)),
    1.96 * np.sqrt(np.diag(K)),
    color="C0",
    alpha=0.1,
)
ax0.axes.xaxis.set_ticks([])
ax0.set_title("Matern")
ax0.set_ylim(-3, 3)
ax0.set_xlim(-5, 5)
kernel = gpflow.kernels.RBF()
K = kernel(xx)
f_samples = np.random.multivariate_normal(np.zeros(100), K, size=5)
ax1.plot(xx, f_samples.T, "C0", linewidth=0.5)
ax1.fill_between(
    xx[:,0],
    -1.96 * np.sqrt(np.diag(K)),
    1.96 * np.sqrt(np.diag(K)),
    color="C0",
    alpha=0.1,
)
ax1.axes.xaxis.set_ticks([])
ax1.set_title("RBF")
ax1.set_ylim(-3, 3)
ax1.set_xlim(-5, 5)
kernel = gpflow.kernels.Periodic(kernel)
K = kernel(xx)
f_samples = np.random.multivariate_normal(np.zeros(100), K, size=5)
ax2.plot(xx, f_samples.T, "C0", linewidth=0.5)
ax2.fill_between(
    xx[:,0],
    -1.96 * np.sqrt(np.diag(K)),
    1.96 * np.sqrt(np.diag(K)),
    color="C0",
    alpha=0.1,
)
ax2.set_title("Periodic")
ax2.set_ylim(-3, 3)
ax2.set_xlim(-5, 5)
plt.tight_layout()
plt.savefig("prior.pdf")



# %%
