# %%
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import zero_mean, get_MOI, get_MOI2D, get_Pendulum_Invariance, get_SHM_Invariance, get_SHM2D_Invariance, get_Double_Pendulum_Invariance
from invariance_functions import degree_of_freedom, get_GPR_2Dmodel, get_SHM2D_data, evaluate_2Dmodel, get_double_pendulum_data
from local_invariance_kernels import  get_Polynomial_2D_Local_Invariance
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# %%
mean = zero_mean(4)

time_step = 0.1
training_time = 1
testing_time = 1

max_angle = 60
n_train = 10 
train_starting_position1 = np.random.uniform(0, max_angle, (n_train))
train_starting_position2 = np.random.uniform(0, max_angle, (n_train))
train_starting_velocity1 = np.random.uniform(0/3, max_angle/3, (n_train))
train_starting_velocity2 = np.random.uniform(0/3, max_angle/3, (n_train))
test_starting_position1 = np.random.uniform(0, max_angle)
test_starting_position2 = np.random.uniform(0, max_angle)
test_starting_velocity1 = np.random.uniform(0/3, max_angle/3)
test_starting_velocity2 = np.random.uniform(0/3, max_angle/3)

print(test_starting_position1)
print(test_starting_position2)
print(test_starting_velocity1)
print(test_starting_velocity2)
data = get_double_pendulum_data(time_step, training_time, 1e-8, train_starting_position1, train_starting_position2, train_starting_velocity1, train_starting_velocity2) #switch
test_data = get_double_pendulum_data(time_step, testing_time, 1e-8,[test_starting_position1],[test_starting_position2],[test_starting_velocity1],[test_starting_velocity2] )
for jitter in [2e-5]:
#    print("current jitter %s" %jitter)
    m = get_GPR_2Dmodel(get_MOI2D(), mean, data, 100)
    print("%s, "%round(m.log_marginal_likelihood().numpy()))
    evaluate_moi = evaluate_2Dmodel(m, test_data, time_step)
    print(evaluate_moi[:2])
    for invar_density in [5]: #np.arange(10, 30, 10):
            try:
                kernel = get_Polynomial_2D_Local_Invariance(2, 40, jitter, [2,2,2,2]) #switch
#                kernel = get_Double_Pendulum_Invariance(3, invar_density, jitter) #switch
                m = get_GPR_2Dmodel(kernel, mean, data, 500)
                print(round(m.log_marginal_likelihood().numpy()))
                evaluate_invariance = evaluate_2Dmodel(m, test_data, time_step)
                print(evaluate_invariance[:2])

            except tf.errors.InvalidArgumentError:
                print("jitter too small")
                break 

# %%
