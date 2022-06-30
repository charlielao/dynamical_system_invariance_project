# %%
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import get_MOI, zero_mean
from invariance_functions import degree_of_freedom, get_GPR_model, get_damped_SHM_data, get_damped_pendulum_data, get_grid_of_points, evaluate_model
from damped_invariance_kernels import get_Damped_Pendulum_Invariance, get_Damped_SHM_Invariance
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
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

for gamma in [0.01, 0.05, 0.1]:
    print("current damping: %s" %gamma)
    data = get_damped_SHM_data(gamma, time_step, training_time, 1e-8, train_starting_position, train_starting_velocity) #switch
    test_data = get_damped_SHM_data(gamma, time_step, testing_time, 1e-8, [test_starting_position], [test_starting_velocity])
    mean_function = zero_mean(2)
    for jitter in [1e-5]:
        m = get_GPR_model(get_MOI(), mean_function, data, 100)
        print("%s, "%round(m.log_marginal_likelihood().numpy()))
        print(evaluate_model(m, test_data, time_step)[:2])
        for invar_density in [40]:#np.arange(10, 40, 10):
                try:
                    kernel = get_Damped_SHM_Invariance(3, invar_density, jitter)#switch
                    m = get_GPR_model(kernel, mean, data, 100)
                    print(round(m.log_marginal_likelihood().numpy()))
                    evaluate_invariance = evaluate_model(m, test_data, time_step)
                    print(m.kernel.epsilon.numpy())
                    print(evaluate_invariance[:2])

                except tf.errors.InvalidArgumentError:
                    print("jitter too small")
                    break

# %%
