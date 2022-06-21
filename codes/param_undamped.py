import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_mean_functions import zero_mean
from parameterised_invariance_kernels import get_Polynomial_Invariance
from invariance_kernels import get_MOI
from invariance_functions import degree_of_freedom, get_GPR_model, get_SHM_data, get_pendulum_data, get_grid_of_points
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

test_grids = get_grid_of_points(3, 40)
zero_mean = zero_mean(2)
data = get_pendulum_data(0.5, 0.1, [90, 130]) #switch
for jitter in [1e-5]:
    print("current jitter %s" %jitter)
    print("Naive GP            lml: %s" %get_GPR_model(get_MOI(), zero_mean, data, test_grids, 100)[0].log_marginal_likelihood().numpy())
#    print("%s, "%round(get_GPR_model(get_MOI(), zero_mean, data, test_grids)[0].log_marginal_likelihood().numpy()))
    for invar_density in [20]: #np.arange(10, 30, 10):
        for poly_d in [4, 5, 6, 7, 8, 9]:
            try:
                kernel = get_Polynomial_Invariance(3, invar_density, jitter, poly_d) #switch
                m, pred, var = get_GPR_model(kernel, zero_mean, data, test_grids, 300)
                print("Invariance GP  %s degrees lml: %s" %(poly_d, m.log_marginal_likelihood().numpy()))
                print(kernel.f_poly.numpy())
                print(kernel.g_poly.numpy())
#                print(round(m.log_marginal_likelihood().numpy()))

            except tf.errors.InvalidArgumentError:
                print("jitter too small")
                break 
