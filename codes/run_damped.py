# %%
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import get_MOI, get_Pendulum_Invariance, get_SHM_Invariance, zero_mean
from invariance_functions import degree_of_freedom, get_GPR_model, get_damped_SHM_data, get_damped_pendulum_data, get_grid_of_points, evaluate_model
from damped_invariance_kernels import get_Damped_Pendulum_Invariance, get_Damped_SHM_Invariance
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

for gamma in [0.01, 0.05, 0.1]:
    print("current damping: %s" %gamma)
#    data = get_damped_pendulum_data(gamma, 1, 0.01, [90, 130]) #switch
    data = get_damped_SHM_data(gamma, 0.1, 3, 1e-2,  [1, 2], [0,0]) #switch
    test_data = get_damped_SHM_data(gamma, 0.1, 1, 1e-2, [3], [1])

    mean_function = zero_mean(2)
    for jitter in [1e-5]:
        m = get_GPR_model(get_MOI(), mean_function, data, 100)
        print("%s, "%round(m.log_marginal_likelihood().numpy()))
        print(evaluate_model(m, test_data, 0.1)[:2])
        for invar_density in [40]:#np.arange(10, 40, 10):
                try:
                    kernel = get_Damped_SHM_Invariance(3, invar_density, jitter)#switch

                    m = get_GPR_model(kernel, mean_function, data, 200)
                    print(round(m.log_marginal_likelihood().numpy()))
                    print(m.kernel.epsilon.numpy())
                    print(evaluate_model(m, test_data, 0.1)[:2])

                except tf.errors.InvalidArgumentError:
                    print("jitter too small")
                    break

# %%
