
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import ZeroMean , get_MOI, get_MOI_2D, get_SHM_invariance_2D 
from invariance_functions import degree_of_freedom, get_GPR_model_2D, get_SHM_data_2D, get_double_pendulum_data, evaluate_model_future_2D, get_GPR_model_sparse_2D, evaluate_model_grid_2D, SHM_dynamics1_2D, SHM_dynamics2_2D
from local_invariance_kernels import  get_polynomial_local_invariance_2D, get_SHM_local_invariance_2D, get_double_pendulum_local_invariance
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
mean = ZeroMean(4) 

time_step = 0.01
training_time = 0.1
testing_time = 0.5

max_x = 5
n_train = 3
train_starting_position1 = np.random.uniform(-max_x, max_x, (n_train))
train_starting_position2 = np.random.uniform(-max_x, max_x, (n_train))
train_starting_velocity1 = np.random.uniform(-max_x/5, max_x/5, (n_train))
train_starting_velocity2 = np.random.uniform(-max_x/5, max_x/5, (n_train))

print(train_starting_position1)
print(train_starting_position2)
print(train_starting_velocity1)
print(train_starting_velocity2)

data = get_SHM_data_2D(time_step, training_time, 1e-8, train_starting_position1, train_starting_position2, train_starting_velocity1, train_starting_velocity2) #switch

test_starting_position1 = np.random.uniform(-max_x, max_x)
test_starting_position2 = np.random.uniform(-max_x, max_x)
test_starting_velocity1 = np.random.uniform(-max_x/5, max_x/5)
test_starting_velocity2 = np.random.uniform(-max_x/5, max_x/5)

print(test_starting_position1)
print(test_starting_position2)
print(test_starting_velocity1)
print(test_starting_velocity2)

jitter = 1e-5
n_neighbours = 20
print("moi")
moi = get_GPR_model_2D(get_MOI_2D(), mean, data, "scipy", 1000, 0.1)
evaluate_moi = evaluate_model_future_2D(moi, test_starting_position1, test_starting_position2, test_starting_velocity1, test_starting_velocity2, SHM_dynamics1_2D, SHM_dynamics2_2D, testing_time, time_step)
print("MOI: %s"%round(moi.log_marginal_likelihood().numpy()))
print(evaluate_moi[0])
try:
    print("known")
    kernel = get_SHM_invariance_2D(5, 6, jitter) #switch
    known = get_GPR_model_2D(kernel, mean, data, "scipy", iterations=1000, lr=0.1)
    evaluate_known = evaluate_model_future_2D(known, test_starting_position1, test_starting_position2, test_starting_velocity1, test_starting_velocity2, SHM_dynamics1_2D, SHM_dynamics2_2D, testing_time, time_step)
    print("Known: %s"%round(known.log_marginal_likelihood().numpy()))
    print(evaluate_known[0])
    print("known local")
    kernel_local = get_SHM_local_invariance_2D(0.5, 5, n_neighbours, jitter) #switch
    known_local = get_GPR_model_2D(kernel_local, mean, data, "scipy", iterations=1000, lr=0.1)
    evaluate_known_local = evaluate_model_future_2D(known_local, test_starting_position1, test_starting_position2, test_starting_velocity1, test_starting_velocity2, SHM_dynamics1_2D, SHM_dynamics2_2D, testing_time, time_step)
    print("known_local: %s"%round(known_local.log_marginal_likelihood().numpy()))
    print(evaluate_known_local[0])
    polynomial_degree = 1
    print("polynomial local")
    kernel = get_polynomial_local_invariance_2D(0.5, 5, n_neighbours, jitter, 1) #switch
#    kernel = get_polynomial_invariance_2D(3, 6, jitter, [polynomial_degree,polynomial_degree,polynomial_degree,polynomial_degree]) #switch
    m, best = get_GPR_model_sparse_2D(kernel, mean, data, "adam", iterations=30000, lr=0.001, reg=0, drop_rate=0)
    evaluate_invariance = evaluate_model_future_2D(m, test_starting_position1,test_starting_position2, test_starting_velocity1, test_starting_velocity2, SHM_dynamics1_2D, SHM_dynamics2_2D, testing_time , time_step)
    print(kernel.poly)
    print("Learnt: %s"%round(m.log_marginal_likelihood().numpy()))
    print(evaluate_invariance[0])
    '''
    print("MOI: %s"%round(moi.log_marginal_likelihood().numpy()))
    print(evaluate_moi[0])
    print("Known: %s"%round(known.log_marginal_likelihood().numpy()))
    print(evaluate_known[0])
    '''
except tf.errors.InvalidArgumentError:
    print("jitter too small")