import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import get_MOI, get_Pendulum_Invariance, get_SHM_Invariance, zero_mean, damping_SHM_mean, damping_pendulum_mean
from invariance_functions import degree_of_freedom, get_GPR_model, get_damped_SHM_data, get_damped_pendulum_data, get_grid_of_points
from damped_invariance_kernels import get_Damped_Pendulum_Invariance, get_Damped_SHM_Invariance
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

test_grids = get_grid_of_points(3, 40)
for gamma in [0.01, 0.05, 0.1]:
    print("current damping: %s" %gamma)
    data = get_damped_SHM_data(gamma, 1, 0.1)#), [90, 130]) #switch
    mean_function = zero_mean(2)
    for jitter in [1e-5]:
#            print("current jitter %s" %jitter)
#            print("Naive GP            lml: %s" %get_GPR_model(get_MOI(), zero_mean(2), data, test_grids)[0].log_marginal_likelihood().numpy())
        print("%s, " %round(get_GPR_model(get_MOI(), mean_function , data, test_grids, 100)[0].log_marginal_likelihood().numpy()))
        for invar_density in [20]:#np.arange(10, 40, 10):
                try:
                    kernel = get_Pendulum_Invariance(3, invar_density, jitter)#switch
                    m, pred, var = get_GPR_model(kernel, mean_function, data, test_grids, 200)
#                        print("Invariance GP density %s lml: %s" %(invar_density, m.log_marginal_likelihood().numpy()))
                    print(round(m.log_marginal_likelihood().numpy()))

                except tf.errors.InvalidArgumentError:
                    print("jitter too small")
                    break
