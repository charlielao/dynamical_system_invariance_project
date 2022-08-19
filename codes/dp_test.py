
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from invariance_kernels import ZeroMean , get_MOI, get_MOI_2D 
from invariance_functions import degree_of_freedom, get_GPR_model_2D, get_SHM_data_2D, get_double_pendulum_data, evaluate_loaded_model_future_2D, evaluate_model_grid_2D, SHM_dynamics1_2D, SHM_dynamics2_2D, get_GPR_model_GD_2D, double_pendulum_dynamics1, double_pendulum_dynamics2
from local_invariance_kernels import  get_polynomial_local_invariance_2D, get_SHM_local_invariance_2D, get_double_pendulum_local_invariance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

mean = ZeroMean(4) 

time_step = 0.01
training_time = 0.1
testing_time = 1

max_x = 60
max_v = 10 
n_train = 5

scalerX = joblib.load('double_pendulum/scalerX.gz')
scalerY = joblib.load('double_pendulum/scalerY.gz')
moi = tf.saved_model.load("double_pendulum/moi")
known = tf.saved_model.load("double_pendulum/known")
model = tf.saved_model.load("double_pendulum/model")

#loaded_result = loaded_model.predict_f_compiled(samples_input)

scalers = (scalerX, scalerY)
time_setting = (testing_time, time_step)
dynamics = (double_pendulum_dynamics1, double_pendulum_dynamics2)
jitter = 1e-4



eva_future_moi = []
eva_future_known = []
eva_future_learnt = []

def energy(X):
    return -2*np.cos(X[:,0])-np.cos(X[:,1])+0.5*tf.square(X[:,2])+0.5*(tf.square(X[:,2])+tf.square(X[:,3])+2*X[:,2]*X[:,3]*np.cos(X[:,0]-X[:,1]))
#lml_moi = moi.log_marginal_likelihood().numpy()
#lml_inv = known.log_marginal_likelihood().numpy()
#lml_inv_p = model.log_marginal_likelihood().numpy()
for i in range(5):
    print(i)
    test_starting_position1 = np.radians(np.random.uniform(-max_x, max_x))
    test_starting_position2 = np.radians(np.random.uniform(-max_x, max_x))
    test_starting_velocity1 = np.radians(np.random.uniform(-max_v, max_v))
    test_starting_velocity2 = np.radians(np.random.uniform(-max_v, max_v))
    test_starting = (test_starting_position1, test_starting_position2, test_starting_velocity1, test_starting_velocity2)

    evaluate_moi = evaluate_loaded_model_future_2D(moi, test_starting, dynamics, time_setting, scalers, energy)
    eva_future_moi.append(evaluate_moi[0])
    print(evaluate_moi[0])
    evaluate_known = evaluate_loaded_model_future_2D(known, test_starting, dynamics, time_setting, scalers, energy)
    eva_future_known.append(evaluate_known[0])
    print(evaluate_known[0])
    evaluate_learnt = evaluate_loaded_model_future_2D(model, test_starting, dynamics, time_setting, scalers, energy)
    eva_future_learnt.append(evaluate_learnt[0])
    print(evaluate_learnt[0])

#print("Log Marginal Likelihood & " + format((lml_moi),".2f")+" & "+ format((lml_inv),".2f") + " & "+ format((lml_inv_p),".2f") + " & "+' \\\\')
print("MSE & "+ format((np.mean(eva_future_moi)),".4f")+" & "+format(np.mean(eva_future_known),".4f")+" & "+format((np.mean(eva_future_learnt)),".4f")+ ' \\\\')

import matplotlib.pyplot as plt
t = np.linspace(0, testing_time, int(testing_time/time_step))
fig, axs = plt.subplots(2,2)
axs[0][0].plot(t,evaluate_known[4][:,0],label="truth", color="black")
axs[0][0].plot(t,evaluate_moi[1][:,0], "--", label="RBF", color="red")
axs[0][0].plot(t,evaluate_known[1][:,0], "--", label="known", color="blue")
axs[0][0].plot(t,evaluate_learnt[1][:,0], "--", label="learnt", color="green")
axs[0][0].set_title("q_1")
axs[0][0].set_xlabel("t")
axs[0][0].legend()

axs[0][1].plot(t,evaluate_known[4][:,1],label="truth", color="black")
axs[0][1].plot(t,evaluate_moi[1][:,1], "--", label="RBF", color="red")
axs[0][1].plot(t,evaluate_known[1][:,1], "--", label="known", color="blue")
axs[0][1].plot(t,evaluate_learnt[1][:,1], "--", label="learnt", color="green")
axs[0][1].set_title("q_2")
axs[0][1].set_xlabel("t")
axs[0][1].legend()

axs[1][0].plot(t,evaluate_known[4][:,2],label="truth", color="black")
axs[1][0].plot(t,evaluate_moi[1][:,2], "--", label="RBF", color="red")
axs[1][0].plot(t,evaluate_known[1][:,2], "--", label="known", color="blue")
axs[1][0].plot(t,evaluate_learnt[1][:,2], "--", label="learnt", color="green")
axs[1][0].set_title("p_1")
axs[1][0].set_xlabel("t")
axs[1][0].legend()

axs[1][1].plot(t,evaluate_known[4][:,3],label="truth", color="black")
axs[1][1].plot(t,evaluate_moi[1][:,3], "--", label="RBF", color="red")
axs[1][1].plot(t,evaluate_known[1][:,3], "--", label="known", color="blue")
axs[1][1].plot(t,evaluate_learnt[1][:,3], "--", label="learnt", color="green")
axs[1][1].set_title("p_2")
axs[1][1].set_xlabel("t")
axs[1][1].legend()

plt.subplots_adjust(left=0.2,
                    bottom=0.2, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
fig.tight_layout()
plt.savefig("figures/double_pendulum_predicted.pdf")


fig, ax = plt.subplots()
plt.plot(t, evaluate_learnt[5], "--",label="true", color="black")
plt.plot(t, evaluate_moi[6], "--",label="RBF", color="red")
plt.plot(t, evaluate_known[6], "--",label="known", color="blue")
plt.plot(t, evaluate_learnt[6], "--",label="learnt", color="green")
plt.legend()
plt.xlabel("t")
plt.ylabel("E")
plt.savefig("figures/double_pendulum_energy.pdf")

n_neighbours = 30
polynomial_degree = 3


print("learnt")
test_starting_position1 = np.radians(np.random.uniform(-max_x, max_x))
test_starting_position2 = np.radians(np.random.uniform(-max_x, max_x))
test_starting_velocity1 = np.radians(np.random.uniform(-max_v, max_v))
test_starting_velocity2 = np.radians(np.random.uniform(-max_v, max_v))
test_starting = (test_starting_position1, test_starting_position2, test_starting_velocity1, test_starting_velocity2)

evaluate_moi = evaluate_model_future_2D(moi, test_starting, dynamics, time_setting, scalers, energy)
print(evaluate_moi[0])
evaluate_known = evaluate_model_future_2D(known, test_starting, dynamics, time_setting, scalers, energy)
print(evaluate_known[0])
evaluate_learnt = evaluate_model_future_2D(model, test_starting, dynamics, time_setting, scalers, energy)
print(evaluate_learnt[0])


grids_lml = []
grids_eva = []
for i in range(5):
    print(i)
    kernel_grid = get_polynomial_local_invariance_2D(1.5, 6, 0, 0.5, n_neighbours, jitter, polynomial_degree) 
    kernel_grid.poly = gpflow.Parameter(0.1*np.random.normal(size=kernel_grid.poly.shape), transform =tfp.bijectors.Sigmoid(to_default_float(-1.), to_default_float(1.)), trainable=False, prior=tfp.distributions.Laplace(to_default_float(0),(0.1)), name="poly")
    model_grid = get_GPR_model_2D(kernel_grid, mean, data, iterations=1000, old_model=known, fixed=True)
    print(model_grid.log_marginal_likelihood().numpy())

    evaluate_learnt_grid = evaluate_model_future_2D(model_grid, test_starting, dynamics, time_setting, scalers, energy)
    print(evaluate_learnt_grid[0])
    grids_eva.append(evaluate_learnt_grid[0])
    grids_lml.append(model_grid.log_marginal_likelihood().numpy())

kernel_free = get_polynomial_local_invariance_2D(1.5, 6, 0, 0.5, n_neighbours, jitter, polynomial_degree) 
model_free = get_GPR_model_2D(kernel_free, mean, data, iterations=1000, old_model=known)
evaluate_learnt_free = evaluate_model_future_2D(model_free, test_starting, dynamics, time_setting, scalers, energy)

plt.scatter(grids_lml, grids_eva, s=10, label="randomly initialised polynomial")
plt.scatter(model.log_marginal_likelihood().numpy(), evaluate_learnt[0],marker="X",s=10, color="red", label="theoretically initialised polynomial", alpha=0.5)
plt.scatter(model_free.log_marginal_likelihood().numpy(), evaluate_learnt_free[0], s=10, color="green", label="optimised", alpha=0.5)
plt.xlabel("marginal likelihood")
plt.ylabel("MSE")
plt.legend()
plt.savefig("figures/double_pendulum_polynomial.pdf")

from scipy.stats import pearsonr
pearsonr(grids_lml, grids_eva)

