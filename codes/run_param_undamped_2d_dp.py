
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import ZeroMean , get_MOI, get_MOI_2D, get_SHM_invariance_2D 
from invariance_functions import degree_of_freedom, get_GPR_model_2D, get_SHM_data_2D, get_double_pendulum_data, evaluate_model_future_2D, evaluate_model_grid_2D, SHM_dynamics1_2D, SHM_dynamics2_2D, get_GPR_model_GD_2D, double_pendulum_dynamics1, double_pendulum_dynamics2
from local_invariance_kernels import  get_polynomial_local_invariance_2D, get_SHM_local_invariance_2D, get_double_pendulum_local_invariance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
mean = ZeroMean(4) 

time_step = 0.01
training_time = 0.1
testing_time = 3

max_x = 60
max_v = 10
n_train = 5
train_starting_position1 = np.random.uniform(-max_x, max_x, (n_train))
train_starting_position2 = np.random.uniform(-max_x, max_x, (n_train))
train_starting_velocity1 = np.random.uniform(-max_v, max_v, (n_train))
train_starting_velocity2 = np.random.uniform(-max_v, max_v, (n_train))

print(train_starting_position1)
print(train_starting_position2)
print(train_starting_velocity1)
print(train_starting_velocity2)

data2 = get_double_pendulum_data(time_step, training_time, 1e-8, train_starting_position1, train_starting_position2, train_starting_velocity1, train_starting_velocity2) #switch

#scalerX = StandardScaler(with_mean=False, with_std=False).fit(data2[0])
#scalerY = StandardScaler(with_mean=False, with_std=False).fit(data2[1])
scalerX = MinMaxScaler((-1,1)).fit(data2[0])
scalerY = MinMaxScaler((-1,1)).fit(data2[1])
X = scalerX.transform(data2[0])
Y = scalerY.transform(data2[1])
data = (X, Y)
scalers = (scalerX, scalerY)
time_setting = (testing_time, time_step)
dynamics = (double_pendulum_dynamics1, double_pendulum_dynamics2)
jitter = 1e-5

eva_future_moi = []
eva_future_inv = []
eva_future_inv_p = []

print("moi")
moi = get_GPR_model_2D(get_MOI_2D(), mean, data, 100)
print(moi.log_marginal_likelihood().numpy())
try:
    n_neighbours = 40
    print("known")
    kernel = get_double_pendulum_local_invariance(0.1, 1, n_neighbours, jitter) #switch
    known = get_GPR_model_2D(kernel, mean, data, iterations=1000)
    print(known.log_marginal_likelihood().numpy())

    polynomial_degree = 3
    print("polynomial local")
    kernel = get_polynomial_local_invariance_2D(0.1, 1, n_neighbours, jitter, polynomial_degree ) #switch
    m = get_GPR_model_GD_2D(kernel, mean, data, iterations=30000, lr=0.001)
    print(m.log_marginal_likelihood().numpy())

    moi.predict_f_compiled = tf.function(moi.predict_f)
    known.predict_f_compiled = tf.function(known.predict_f)
    m.predict_f_compiled = tf.function(m.predict_f)

    samples_input = tf.convert_to_tensor(data[0])
    moi.predict_f_compiled(samples_input)
    known.predict_f_compiled(samples_input)
    m.predict_f_compiled(samples_input)

    tf.saved_model.save(moi, "double_pendulum_model/moi")
    tf.saved_model.save(known, "double_pendulum_model/known")
    tf.saved_model.save(m, "double_pendulum_model/m")

    '''
    #loaded_moi = tf.saved_model.load("shm_2d_model/moi")
    #loaded_known = tf.saved_model.load("shm_2d_model/known")
    #loaded_m = tf.saved_model.load("shm_2d_model/m")
    #loaded_result = loaded_model.predict_f_compiled(samples_input)
    for i in range(3):
        print(i)
        test_starting_position1 = np.random.uniform(-max_x, max_x)
        test_starting_position2 = np.random.uniform(-max_x, max_x)
        test_starting_velocity1 = np.random.uniform(-max_x/5, max_x/5)
        test_starting_velocity2 = np.random.uniform(-max_x/5, max_x/5)

        test_starting = (test_starting_position1, test_starting_position2, test_starting_velocity1, test_starting_velocity2)
        print(test_starting)
        evaluate_moi = evaluate_model_future_2D(moi, test_starting, dynamics, time_setting, scalers)
        eva_future_moi.append(evaluate_moi[0])
        evaluate_invariance = evaluate_model_future_2D(m, test_starting, dynamics, time_setting, scalers)
        eva_future_inv.append(evaluate_invariance[0])
        if i==2:
        #    eva = evaluate_model_future_2D(m, test_starting, dynamics, time_setting, scalers, (kernel.inv_f1, kernel.inv_f2, kernel.inv_g1, kernel.inv_g2), (lambda x: x[2], lambda x: x[3], lambda x:x[0], lambda x:x[1]))
        #    eva = evaluate_model_future_2D(m, test_starting, dynamics, time_setting, scalers, (kernel.inv_f1, kernel.inv_f2, kernel.inv_g1, kernel.inv_g2), (lambda x: 2*x[2]+x[3]*np.cos(x[0]-x[1]), lambda x: x[3]+x[2]*np.cos(x[0]-x[1]), lambda x:2*np.sin(x[0])-x[2]*x[3]*np.sin(x[0]-x[1]), lambda x:np.sin(x[1])+x[2]*x[3]*np.sin(x[0]-x[1])))
        else:
            evaluate_invariance_p = evaluate_model_future_2D(m, test_starting, dynamics, time_setting, scalers)
        eva_future_inv_p.append(evaluate_invariance_p[0])
        print(evaluate_invariance[0])

    print("Baseline RBF & " + format((lml_moi),".2f")+" & "+format(np.mean(eva_future_moi),".4f") +' \\\\')
    print("Invariance Kernel & "+ format((lml_inv),".2f")+" & "+format(np.mean(eva_future_inv),".4f")+ ' \\\\')

    '''

#    plt.plot(eva[5])
#    plt.plot(eva[6])
#    plt.show()
#    plt.savefig("test.pdf")
except tf.errors.InvalidArgumentError:
    print("jitter too small")
