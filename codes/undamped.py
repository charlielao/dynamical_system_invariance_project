import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_mean_functions import zero_mean
from invariance_kernels import get_MOI, get_Pendulum_Invariance, get_SHM_Invariance
from invariance_functions import degree_of_freedom, get_GPR_model, get_SHM_data, get_pendulum_data, get_grid_of_points
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

test_grids = get_grid_of_points(3, 40)
zero_mean = zero_mean(2)
data = get_SHM_data(30, 0.01) #switch
for jitter in [1e-5]:
    print("current jitter %s" %jitter)
    print("Naive GP            lml: %s" %get_GPR_model(get_MOI(), zero_mean, data, test_grids)[0].log_marginal_likelihood().numpy())
    for invar_density in [20]: #np.arange(10, 30, 10):
            try:
                kernel = get_SHM_Invariance(3, invar_density, jitter) #switch
                m, pred, var = get_GPR_model(kernel, zero_mean, data, test_grids)
                print("Invariance GP density %s lml: %s" %(invar_density, m.log_marginal_likelihood().numpy()))

            except tf.errors.InvalidArgumentError:
                print("jitter too small")
                break 
