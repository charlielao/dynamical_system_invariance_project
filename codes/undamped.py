import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
import invariance_functions as inv
test_grids = inv.get_grid_of_points(3, 40)
zero_mean = inv.zero_mean(2)
print("SHM") #switch
data = inv.get_SHM_data(30, 0.1) #switch
for jitter in [1e-5, 1e-6, 1e-8]:
    print("current jitter %s" %jitter)
    print("Naive GP lml: %s" %inv.get_GPR_model(inv.get_MOI(), zero_mean, data, test_grids)[0].log_marginal_likelihood().numpy())
    for invar_density in [20]:#np.arange(10, 40, 10):
            try:
                kernel = inv.get_SHM_Invariance(3, invar_density, jitter)
                m, pred, var = inv.get_GPR_model(kernel, zero_mean, data, test_grids)
                print("Invariance GP density %s lml: %s" %(invar_density, m.log_marginal_likelihood().numpy()))

            except tf.errors.InvalidArgumentError:
                print("jitter too small")
                continue
    print("\n")
