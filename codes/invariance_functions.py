import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
def degree_of_freedom(kernel, test_points):
    K = kernel(test_points)
    return tf.linalg.trace(tf.tensordot(K, tf.linalg.inv(K+1e-6*tf.eye(K.shape[0], dtype=tf.float64)), 1)).numpy()

def get_SHM_data(time_interval, noise):
    t = tf.linspace(0, 30, int(30/time_interval))
    x = tf.math.sin(t)
    v = tf.math.cos(t) 
    X1 = tf.concat([x[:,None], v[:,None]], axis=-1)
    X2 = 2*X1
    X1 += tf.random.normal((X1.shape), 0, noise, dtype=tf.float64)
    X2 += tf.random.normal((X2.shape), 0, noise, dtype=tf.float64)
    Y1 = (X1[2:,:]-X1[:-2, :])/(2*time_interval) 
    Y2 = (X2[2:,:]-X2[:-2, :])/(2*time_interval) 
    X1 = X1[1:-1, :]
    X2 = X2[1:-1, :]
    X = tf.concat([X1, X2], axis=0)
    Y = tf.concat([Y1, Y2], axis=0) 
    return (X, Y)

def get_damped_SHM_data(gamma, time_interval, noise):
    m = 1
    k = 1
    w = np.sqrt(k/m-gamma**2)
    t = np.linspace(0, 30, int(30/time_interval))
    x = np.sin(w*t)*np.exp(-gamma*t)
    v = np.exp(-gamma*t)*(w*np.cos(w*t)-gamma*np.sin(w*t))
    X1 = tf.concat([x[:,None], v[:,None]], axis=-1)
    X2 = 2*X1
    X1 += tf.random.normal((X1.shape), 0, noise, dtype=tf.float64)
    X2 += tf.random.normal((X2.shape), 0, noise, dtype=tf.float64)
    Y1 = (X1[2:,:]-X1[:-2, :])/(2*time_interval) 
    Y2 = (X2[2:,:]-X2[:-2, :])/(2*time_interval) 
    X1 = X1[1:-1, :]
    X2 = X2[1:-1, :]
    X = tf.concat([X1, X2], axis=0)
    Y = tf.concat([Y1, Y2], axis=0)
    return (X, Y)

def get_pendulum_data(time_interval, noise):
    dt = 0.01
    sample_rate = int(time_interval/dt)
    t = np.linspace(0, 30, int(30/dt))
    g = 1
    l = 1
    def f(t, r):
        theta = r[0]
        omega = r[1]
        return np.array([omega, -g / l * np.sin(theta)])
    results = odeint(f, [np.radians(90), 0], t, tfirst=True)
    results2 = odeint(f, [np.radians(150), 0], t, tfirst=True)
    x1 = results[0::sample_rate,0]
    v1 = results[0::sample_rate,1]
    x2 = results2[0::sample_rate,0]
    v2 = results2[0::sample_rate,1]
    t = t[0::sample_rate]
    X_1 = tf.concat([x1[:,None], v1[:,None]], axis=-1)
    X_2 = tf.concat([x2[:,None], v2[:,None]], axis=-1)
    X_1 += tf.random.normal((X_1.shape), 0, noise, dtype=tf.float64)
    X_2 += tf.random.normal((X_2.shape), 0, noise, dtype=tf.float64)
    Y_1 = (X_1[2:,:]-X_1[:-2, :])/(2*time_interval) 
    Y_2 = (X_2[2:,:]-X_2[:-2, :])/(2*time_interval) 
    X_1 = X_1[1:-1, :]
    X_2 = X_2[1:-1, :]
    X = tf.concat([X_1,X_2], axis=0)
    Y = tf.concat([Y_1,Y_2], axis=0)
    return (X, Y)

def get_damped_pendulum_data(gamma, time_interval, noise):
    dt = 0.01
    sample_rate = int(time_interval/dt)
    t = np.linspace(0, 30, int(30/dt))
    g = 1
    l = 1
    def f(t, r):
        theta = r[0]
        omega = r[1]
        return np.array([omega, -g / l * np.sin(theta)-2*gamma*omega])
    results = odeint(f, [np.radians(90), 0], t, tfirst=True)
    results2 = odeint(f, [np.radians(150), 0], t, tfirst=True)
    x1 = results[0::sample_rate,0]
    v1 = results[0::sample_rate,1]
    x2 = results2[0::sample_rate,0]
    v2 = results2[0::sample_rate,1]
    t = t[0::sample_rate]
    X_1 = tf.concat([x1[:,None], v1[:,None]], axis=-1)
    X_2 = tf.concat([x2[:,None], v2[:,None]], axis=-1)
    X_1 += tf.random.normal((X_1.shape), 0, noise, dtype=tf.float64)
    X_2 += tf.random.normal((X_2.shape), 0, noise, dtype=tf.float64)
    Y_1 = (X_1[2:,:]-X_1[:-2, :])/(2*time_interval) 
    Y_2 = (X_2[2:,:]-X_2[:-2, :])/(2*time_interval) 
    X_1 = X_1[1:-1, :]
    X_2 = X_2[1:-1, :]
    X = tf.concat([X_1,X_2], axis=0)
    Y = tf.concat([Y_1,Y_2], axis=0)
    return (X, Y)

def get_grid_of_points(grid_range, grid_density):
    grid_xs = tf.linspace(-grid_range,grid_range,grid_density)
    grid_vs = tf.linspace(-grid_range,grid_range,grid_density)
    grid_xx, grid_vv = tf.meshgrid(grid_xs, grid_vs)
    grid_points = tf.stack([tf.reshape(grid_xx,[-1]), tf.reshape(grid_vv,[-1])], axis=1)
    return grid_points

def get_GPR_model(kernel, mean_function, data, test_points):
    X, Y = data
    m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,1,None],Y[:,0,None]],1)),(Y.shape[0]*2,1))), kernel=kernel, mean_function=mean_function)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
    pred, var = m.predict_f(test_points)
    return (m, pred, var)

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

