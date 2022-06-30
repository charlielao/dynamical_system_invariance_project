import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import zero_mean, get_MOI
from invariance_functions import degree_of_freedom, get_GPR_model, get_SHM_data, get_pendulum_data, get_grid_of_points
from parameterised_invariance_kernels import get_Polynomial_Invariance
from local_invariance_kernels import get_Polynomial_Local_Invariance
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
mean = zero_mean(2)
time_step = 0.1
training_time = 1
testing_time = 1

max_angle = 60
n_train = 10 
train_starting_position = np.random.uniform(0, max_angle, (n_train))
train_starting_velocity = np.random.uniform(0/3, max_angle/3, (n_train))

test_starting_position = np.random.uniform(-max_angle, max_angle)
test_starting_velocity = np.random.uniform(-max_angle, max_angle)
print(test_starting_position)
print(test_starting_velocity)

data = get_SHM_data(time_step, training_time, 1e-8, train_starting_position, train_starting_velocity) #switch
test_data = get_SHM_data(time_step, testing_time, 1e-8, [test_starting_position], [test_starting_velocity])

for jitter in [1e-5]:
#    print("current jitter %s" %jitter)
    m = get_GPR_model(get_MOI(), mean, data, 100)
    print("%s, "%round(m.log_marginal_likelihood().numpy()))
    evaluate_moi = evaluate_model(m, test_data, time_step)
    print(evaluate_moi[:2])
    for invar_density in [20]: #np.arange(10, 30, 10):
        for poly_f_d in [5, 6, 7]:
            for poly_g_d in [5, 6, 7]:
                try:
                    kernel = get_Polynomial_Local_Invariance(2, 5, jitter, poly_f_d, poly_g_d) #switch
                    m = get_GPR_model(kernel, mean, data, 100)
                    print(round(m.log_marginal_likelihood().numpy()))
                    evaluate_invariance = evaluate_model(m, test_data, time_step)
                    print(evaluate_invariance[:2])
                    print(kernel.f_poly.numpy())
                    print(kernel.g_poly.numpy())

                except tf.errors.InvalidArgumentError:
                    print("jitter too small")
                    break 
