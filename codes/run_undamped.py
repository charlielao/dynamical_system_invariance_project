# %%
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import zero_mean, get_MOI, get_Pendulum_Invariance, get_SHM_Invariance
from invariance_functions import degree_of_freedom, get_GPR_model, get_SHM_data, get_pendulum_data, get_grid_of_points, evaluate_model
from local_invariance_kernels import get_SHM_Local_Invariance, get_Pendulum_Local_Invariance
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# %%
test_data = get_pendulum_data(0.1, 1, 1e-8, [160], [0])
mean = zero_mean(2)
data = get_pendulum_data(0.1, 2, 1e-8, [60, 120], [0,0]) #switch
#data = get_pendulum_data(1, 0.1) #switch
for jitter in [2e-5]:
#    print("current jitter %s" %jitter)
#    print("Naive GP            lml: %s" %get_GPR_model(get_MOI(), zero_mean, data, test_grids)[0].log_marginal_likelihood().numpy())
    m = get_GPR_model(get_MOI(), mean, data, 100)
    print("%s, "%round(m.log_marginal_likelihood().numpy()))
    evaluate_moi = evaluate_model(m, test_data, 0.1)
    print(evaluate_moi[:2])
    for invar_density in [40]: #np.arange(10, 30, 10):
            try:
                kernel = get_Pendulum_Invariance(3, invar_density, jitter) #switch
                m = get_GPR_model(kernel, mean, data, 100)
#                print("Invariance GP density %s lml: %s" %(invar_density, m.log_marginal_likelihood().numpy()))
                print(round(m.log_marginal_likelihood().numpy()))
                evaluate_invariance = evaluate_model(m, test_data, 0.1)
                print(evaluate_invariance[:2])

            except tf.errors.InvalidArgumentError:
                print("jitter too small")
                break 

# %%
