# %%
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
def degree_of_freedom(kernel, test_points):
    K = kernel(test_points)
    return tf.linalg.trace(tf.tensordot(K, tf.linalg.inv(K+1e-6*tf.eye(K.shape[0], dtype=tf.float64)), 1)).numpy()

def get_SHM_data(time_step, total_time, noise, initial_positions):
    m = k = 1
    euler_dt = 0.01
    sample_rate = int(time_step/euler_dt)
    t = tf.linspace(0, total_time, int(total_time/euler_dt))
    x1 = np.zeros(int(total_time/euler_dt))
    x2 = np.zeros(int(total_time/euler_dt))
    v1 = np.zeros(int(total_time/euler_dt))
    v2 = np.zeros(int(total_time/euler_dt))
    x1[0] = initial_positions[0]
    x2[0] = initial_positions[1]
    for i in range(1, int(total_time/euler_dt)):
        x1[i] = x1[i-1] + (v1[i-1]+np.random.normal(0, noise)) * euler_dt
        v1[i] = v1[i-1] + (-k/m*x1[i-1]+np.random.normal(0, noise)) * euler_dt
        x2[i] = x2[i-1] + (v2[i-1]+np.random.normal(0, noise)) * euler_dt
        v2[i] = v2[i-1] + (-k/m*x2[i-1]+np.random.normal(0, noise)) * euler_dt
    x1 = x1[0::sample_rate]
    x2 = x2[0::sample_rate]
    v1 = v1[0::sample_rate]
    v2 = v2[0::sample_rate]
    t = t[0::sample_rate]

    X1 = tf.concat([x1[:,None], v1[:,None]], axis=-1)
    X2 = tf.concat([x2[:,None], v2[:,None]], axis=-1)
    Y1 = (X1[2:,:]-X1[:-2, :])/(2*time_step) 
    Y2 = (X2[2:,:]-X2[:-2, :])/(2*time_step) 
    X1 = X1[1:-1, :]
    X2 = X2[1:-1, :]
    X = tf.concat([X1, X2], axis=0)
    Y = tf.concat([Y1, Y2], axis=0)
    return (X, Y)
# %%

def get_damped_SHM_data(gamma, time_step, total_time, noise, initial_positions):
    m = k = 1
    w02 = k/m
    euler_dt = 0.01
    sample_rate = int(time_step/euler_dt)
    t = tf.linspace(0, total_time, int(total_time/euler_dt))
    x1 = np.zeros(int(total_time/euler_dt))
    x2 = np.zeros(int(total_time/euler_dt))
    v1 = np.zeros(int(total_time/euler_dt))
    v2 = np.zeros(int(total_time/euler_dt))
    x1[0] = initial_positions[0]
    x2[0] = initial_positions[1]
    for i in range(1, int(total_time/euler_dt)):
        x1[i] = x1[i-1] + (v1[i-1]+np.random.normal(0, noise)) * euler_dt
        v1[i] = v1[i-1] + (-2*gamma*v1[i-1]-w02*x1[i-1]+np.random.normal(0, noise)) * euler_dt
        x2[i] = x2[i-1] + (v2[i-1]+np.random.normal(0, noise)) * euler_dt
        v2[i] = v2[i-1] + (-2*gamma*v2[i-1]-w02*x2[i-1]+np.random.normal(0, noise)) * euler_dt
    x1 = x1[0::sample_rate]
    x2 = x2[0::sample_rate]
    v1 = v1[0::sample_rate]
    v2 = v2[0::sample_rate]
    t = t[0::sample_rate]

    X1 = tf.concat([x1[:,None], v1[:,None]], axis=-1)
    X2 = tf.concat([x2[:,None], v2[:,None]], axis=-1)
    Y1 = (X1[2:,:]-X1[:-2, :])/(2*time_step) 
    Y2 = (X2[2:,:]-X2[:-2, :])/(2*time_step) 
    X1 = X1[1:-1, :]
    X2 = X2[1:-1, :]
    X = tf.concat([X1, X2], axis=0)
    Y = tf.concat([Y1, Y2], axis=0)
    return (X, Y)

def get_pendulum_data(time_step, total_time, noise, initial_angles):
    g = l = 1
    euler_dt = 0.01
    sample_rate = int(time_step/euler_dt)
    t = tf.linspace(0, total_time, int(total_time/euler_dt))
    x1 = np.zeros(int(total_time/euler_dt))
    x2 = np.zeros(int(total_time/euler_dt))
    v1 = np.zeros(int(total_time/euler_dt))
    v2 = np.zeros(int(total_time/euler_dt))
    x1[0] = np.radians(initial_angles[0])
    x2[0] = np.raidans(initial_angles[1])
    for i in range(1, int(total_time/euler_dt)):
        x1[i] = x1[i-1] + (v1[i-1]+np.random.normal(0, noise)) * euler_dt
        v1[i] = v1[i-1] + (-g/l*np.sin(x1[i-1])+np.random.normal(0, noise)) * euler_dt
        x2[i] = x2[i-1] + (v2[i-1]+np.random.normal(0, noise)) * euler_dt
        v2[i] = v2[i-1] + (-g/l*np.sin(x2[i-1])+np.random.normal(0, noise)) * euler_dt
    x1 = x1[0::sample_rate]
    x2 = x2[0::sample_rate]
    v1 = v1[0::sample_rate]
    v2 = v2[0::sample_rate]
    t = t[0::sample_rate]

    X1 = tf.concat([x1[:,None], v1[:,None]], axis=-1)
    X2 = tf.concat([x2[:,None], v2[:,None]], axis=-1)
    Y1 = (X1[2:,:]-X1[:-2, :])/(2*time_step) 
    Y2 = (X2[2:,:]-X2[:-2, :])/(2*time_step) 
    X1 = X1[1:-1, :]
    X2 = X2[1:-1, :]
    X = tf.concat([X1, X2], axis=0)
    Y = tf.concat([Y1, Y2], axis=0)
    return (X, Y)

def get_damped_pendulum_data(gamma, time_step, total_time, noise, initial_angles):
    g = l = 1
    w02 = g/l
    euler_dt = 0.01
    sample_rate = int(time_step/euler_dt)
    t = tf.linspace(0, total_time, int(total_time/euler_dt))
    x1 = np.zeros(int(total_time/euler_dt))
    x2 = np.zeros(int(total_time/euler_dt))
    v1 = np.zeros(int(total_time/euler_dt))
    v2 = np.zeros(int(total_time/euler_dt))
    x1[0] = np.radians(initial_angles[0])
    x2[0] = np.radians(initial_angles[1])
    for i in range(1, int(total_time/euler_dt)):
        x1[i] = x1[i-1] + (v1[i-1]+np.random.normal(0, noise)) * euler_dt
        v1[i] = v1[i-1] + (-2*gamma*v1[i-1]-w02*np.sin(x1[i-1])+np.random.normal(0, noise)) * euler_dt
        x2[i] = x2[i-1] + (v2[i-1]+np.random.normal(0, noise)) * euler_dt
        v2[i] = v2[i-1] + (-2*gamma*v2[i-1]-w02*np.sin(x2[i-1])+np.random.normal(0, noise)) * euler_dt
    x1 = x1[0::sample_rate]
    x2 = x2[0::sample_rate]
    v1 = v1[0::sample_rate]
    v2 = v2[0::sample_rate]
    t = t[0::sample_rate]

    X1 = tf.concat([x1[:,None], v1[:,None]], axis=-1)
    X2 = tf.concat([x2[:,None], v2[:,None]], axis=-1)
    Y1 = (X1[2:,:]-X1[:-2, :])/(2*time_step) 
    Y2 = (X2[2:,:]-X2[:-2, :])/(2*time_step) 
    X1 = X1[1:-1, :]
    X2 = X2[1:-1, :]
    X = tf.concat([X1, X2], axis=0)
    Y = tf.concat([Y1, Y2], axis=0)
    return (X, Y)

def get_grid_of_points(grid_range, grid_density):
    grid_xs = tf.linspace(-grid_range,grid_range,grid_density)
    grid_vs = tf.linspace(-grid_range,grid_range,grid_density)
    grid_xx, grid_vv = tf.meshgrid(grid_xs, grid_vs)
    grid_points = tf.stack([tf.reshape(grid_xx,[-1]), tf.reshape(grid_vv,[-1])], axis=1)
    return grid_points

def get_GPR_model(kernel, mean_function, data, test_points, iterations):
    X, Y = data
    m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,1,None],Y[:,0,None]],1)),(Y.shape[0]*2,1))), kernel=kernel, mean_function=mean_function)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=iterations))
    pred, var = m.predict_f(test_points)
    return (m, pred, var)

def evaluate_model(m, ground_truth, time_step):
    X, Y = ground_truth
    predicted = np.zeros(X.shape)
    predicted[0,:] = X[0,:]
    for i in range(1, X.shape[0]):
        predicted[i, :] = predicted[i-1,:] + m.predict_f(predicted[i-1,:])[0]*time_step 
    return predicted 

'''
def plotting(pred, var, test_points, data, save, name, angle1, angle2, acc, lml):
    X, Y = data
    test_xx, test_vv = test_points
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(test_xx, test_vv, tf.reshape(pred, test_xx.shape), cmap="viridis",linewidth=0, antialiased=False, alpha=0.5)
    surf = ax.plot_surface(test_xx, test_vv, tf.reshape(pred, test_xx.shape)+1.96*tf.math.sqrt(tf.reshape(var, test_xx.shape)), color="grey",linewidth=0, antialiased=False, alpha=0.1)
    surf = ax.plot_surface(test_xx, test_vv, tf.reshape(pred, test_xx.shape)-1.96*tf.math.sqrt(tf.reshape(var, test_xx.shape)), color="grey",linewidth=0, antialiased=False, alpha=0.1)
    if acc:
        ax.scatter(X[:,0], X[:,1],Y[:,1,None], color="black", marker="o", s=3) #data
    else:
        ax.scatter(X[:,0], X[:,1],Y[:,0,None], color="black", marker="o", s=3) #data
    ax.view_init(angle1,angle2)
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    if acc:
        ax.set_zlabel("acceleration")
    else:
        ax.set_zlabel("velocity")
    ax.annotate("log marginal likelihood: {0:0.2f}".format(lml), xy=(0.5, 0.9), xycoords='axes fraction')
    if save:
        plt.savefig(name+"_3D.pdf")
    plt.figure(figsize=(5, 3))
    contours = plt.contourf(test_xx, test_vv, tf.reshape(pred,(test_xx.shape)), levels=100, cmap="viridis", alpha=0.3)
    plt.colorbar(contours)
    if acc:
        contours = plt.scatter(X[:,0],X[:,1],c=Y[:,1,None],cmap="viridis", alpha=0.2)
    else:
        contours = plt.scatter(X[:,0],X[:,1],c=Y[:,0,None],cmap="viridis", alpha=0.2)
    plt.xlim((-test_range, test_range))
    plt.ylim((-test_range, test_range))
    plt.xlabel("position")
    plt.ylabel("velocity")
    if save:
        plt.savefig(name+"_contour.pdf")
'''


# %%
