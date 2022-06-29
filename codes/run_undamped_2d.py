# %%
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import zero_mean, get_MOI, get_MOI2D, get_Pendulum_Invariance, get_SHM_Invariance, get_SHM2D_Invariance
from invariance_functions import degree_of_freedom, get_GPR_2Dmodel, get_SHM2D_data, evaluate_2Dmodel
from local_invariance_kernels import get_SHM2D_Local_Invariance
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# %%
mean = zero_mean(4)
data = get_SHM2D_data(0.1, 2, 1e-4, [[1, 2],[0,1]], [[0,0],[0,0]]) #switch
test_data = get_SHM2D_data(0.1, 1, 1e-4, [[np.random.uniform(-3,3)],[np.random.uniform(-3,3)]], [[np.random.uniform(-3,3)],[np.random.uniform(-3,3)]])
for jitter in [2e-5]:
#    print("current jitter %s" %jitter)
    m = get_GPR_2Dmodel(get_MOI2D(), mean, data, 200)
    print("%s, "%round(m.log_marginal_likelihood().numpy()))
    evaluate_moi = evaluate_2Dmodel(m, test_data, 0.1)
    print(evaluate_moi[:2])
    for invar_density in [5]: #np.arange(10, 30, 10):
            try:
                kernel = get_SHM2D_Local_Invariance(3, 50, jitter) #switch
#                kernel = get_SHM2D_Invariance(3, invar_density, jitter) #switch
                m = get_GPR_2Dmodel(kernel, mean, data, 100)
                print(round(m.log_marginal_likelihood().numpy()))
                evaluate_invariance = evaluate_2Dmodel(m, test_data, 0.1)
                print(evaluate_invariance[:2])

            except tf.errors.InvalidArgumentError:
                print("jitter too small")
                break 

# %%
