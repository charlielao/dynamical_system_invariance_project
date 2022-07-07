import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import zero_mean, get_MOI, get_MOI2D, get_Pendulum_Invariance, get_SHM_Invariance, get_SHM2D_Invariance, get_Double_Pendulum_Invariance
from invariance_functions import degree_of_freedom, get_GPR_2Dmodel, get_SHM2D_data, evaluate_2Dmodel, get_double_pendulum_data, get_GPR_2Dmodel_sparse
from local_invariance_kernels import  get_Polynomial_2D_Local_Invariance
from parameterised_invariance_kernels import get_Polynomial_2D_Invariance, get_Polynomial_2D_Invariance_fixed
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
mean = zero_mean(4)

time_step = 0.01
training_time = 0.1
testing_time = 0.1

max_x = 4
n_train = 3
train_starting_position1 = np.random.uniform(-max_x, max_x, (n_train))
train_starting_position2 = np.random.uniform(-max_x, max_x, (n_train))
train_starting_velocity1 = np.random.uniform(-max_x/5, max_x/5, (n_train))
train_starting_velocity2 = np.random.uniform(-max_x/5, max_x/5, (n_train))
test_starting_position1 = np.random.uniform(-max_x, max_x)
test_starting_position2 = np.random.uniform(-max_x, max_x)
test_starting_velocity1 = np.random.uniform(-max_x/5, max_x/5)
test_starting_velocity2 = np.random.uniform(-max_x/5, max_x/5)

print(train_starting_position1)
print(train_starting_position2)
print(train_starting_velocity1)
print(train_starting_velocity2)

print(test_starting_position1)
print(test_starting_position2)
print(test_starting_velocity1)
print(test_starting_velocity2)

data = get_SHM2D_data(time_step, training_time, 1e-8, train_starting_position1, train_starting_position2, train_starting_velocity1, train_starting_velocity2) #switch
test_data = get_SHM2D_data(time_step, testing_time, 1e-8,[test_starting_position1],[test_starting_position2],[test_starting_velocity1],[test_starting_velocity2] )
for jitter in [1e-5]:
    moi = get_GPR_2Dmodel(get_MOI2D(), mean, data, "scipy", 1000, 0.1)
    print("%s, "%round(moi.log_marginal_likelihood().numpy()))
    evaluate_moi = evaluate_2Dmodel(moi, test_data, time_step)
    print(evaluate_moi[:2])
    for invar_density in [8]: #np.arange(10, 30, 10):
            try:
                print("SHM")
                kernel = get_SHM2D_Invariance(4, invar_density, jitter) #switch
                m = get_GPR_2Dmodel(kernel, mean, data, "scipy", iterations=1000, lr=0.1)
                print(round(m.log_marginal_likelihood().numpy()))
                evaluate_invariance = evaluate_2Dmodel(m, test_data, time_step)
                print(evaluate_invariance[:2])
                polynomial_degree = 1
#                kernel = get_Polynomial_2D_Local_Invariance(4, 200, jitter, [polynomial_degree,polynomial_degree,polynomial_degree,polynomial_degree]) #switch
                print("polynomial")
                kernel = get_Polynomial_2D_Invariance(3, invar_density, jitter, [polynomial_degree,polynomial_degree,polynomial_degree,polynomial_degree]) #switch
                m, best = get_GPR_2Dmodel_sparse(kernel, mean, data, "adam", iterations=20000, lr=0.001, reg=10, drop_rate=0)
                print(round(m.log_marginal_likelihood().numpy()))
                evaluate_invariance = evaluate_2Dmodel(m, test_data, time_step)
                print(evaluate_invariance[:2])
                print(kernel.f1_poly)
                print(kernel.f2_poly)
                print(kernel.g1_poly)
                print(kernel.g2_poly)
                

            except tf.errors.InvalidArgumentError:
                print("jitter too small")
                break 
