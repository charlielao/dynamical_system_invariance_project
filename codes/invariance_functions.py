# %%
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
#from sklearn.metrics import mean_squared_error
def degree_of_freedom(kernel, test_points):
    K = kernel(test_points)
    return tf.linalg.trace(tf.tensordot(K, tf.linalg.inv(K+1e-6*tf.eye(K.shape[0], dtype=tf.float64)), 1)).numpy()

def get_SHM_data(time_step, total_time, noise, initial_positions, initial_velocities):
    m = k = 1
    w02 = k/m
    t = tf.linspace(0, total_time, int(total_time/time_step))
    def f(t, r):
        x = r[0]
        v = r[1]
        return np.array([v+np.random.normal(0,noise), -w02*x+np.random.normal(0,noise)])
    initial_conditions = np.array([initial_positions, initial_velocities])
    X = np.apply_along_axis(lambda m: odeint(f, m, t, tfirst=True), 0, initial_conditions)
    Y = (X[2:,:,:] - X[:-2,:,:])/(2*time_step)
    X = X[1:-1,:,:]
    X = X.transpose(2,0,1).reshape(-1,2)
    Y = Y.transpose(2,0,1).reshape(-1,2)
    return (X, Y)
#%%
def get_SHM2D_data(time_step, total_time, noise, initial_positions_1,initial_positions_2, initial_velocities_1,initial_velocities_2):
    m = k = 1
    w02 = k/m
    t = tf.linspace(0, total_time, int(total_time/time_step))
    def f(t, r):
        x1 = r[0]
        x2 = r[1]
        v1 = r[2]
        v2 = r[3]
        return np.array([v1+np.random.normal(0,noise), v2+np.random.normal(0,noise),-w02*x1+np.random.normal(0,noise),-w02*x2+np.random.normal(0,noise)])
    initial_conditions = np.array([initial_positions_1, initial_positions_2, initial_velocities_1, initial_velocities_2])
    X = np.apply_along_axis(lambda m: odeint(f, m, t, tfirst=True), 0, initial_conditions)
    Y = (X[2:,:,:] - X[:-2,:,:])/(2*time_step)
    X = X[1:-1,:,:]
    X = X.transpose(2,0,1).reshape(-1,4)
    Y = Y.transpose(2,0,1).reshape(-1,4)
    return (X, Y)

def get_damped_SHM_data(gamma, time_step, total_time, noise, initial_positions, initial_velocities):
    m = k = 1
    w02 = k/m
    t = tf.linspace(0, total_time, int(total_time/time_step))
    def f(t, r):
        x = r[0]
        v = r[1]
        return np.array([v+np.random.normal(0,noise), -2*gamma*v-w02*x+np.random.normal(0,noise)])
    initial_conditions = np.array([initial_positions, initial_velocities])
    X = np.apply_along_axis(lambda m: odeint(f, m, t, tfirst=True), 0, initial_conditions)
    Y = (X[2:,:,:] - X[:-2,:,:])/(2*time_step)
    X = X[1:-1,:,:]
    X = X.transpose(2,0,1).reshape(-1,2)
    Y = Y.transpose(2,0,1).reshape(-1,2)
    return (X, Y)

def get_pendulum_data(time_step, total_time, noise, initial_angles, initial_angular_velocities):
    g = l = 1
    w02 = g/l
    t = tf.linspace(0, total_time, int(total_time/time_step))
    def f(t, r):
        x = r[0]
        v = r[1]
        return np.array([v+np.random.normal(0,noise), -w02*np.sin(x)+np.random.normal(0,noise)])
    initial_conditions = np.array([initial_angles, initial_angular_velocities])
    X = np.apply_along_axis(lambda m: odeint(f, m, t, tfirst=True), 0, initial_conditions)
    Y = (X[2:,:,:] - X[:-2,:,:])/(2*time_step)
    X = X[1:-1,:,:]
    X = X.transpose(2,0,1).reshape(-1,2)
    Y = Y.transpose(2,0,1).reshape(-1,2)
    return (X, Y)

def get_double_pendulum_data(time_step, total_time, noise, initial_angles, initial_angular_velocities):
    # initial angles/angular velocity will be [[n],[n]]
    g = l1 = l2 = m1 = m2 = 1
    n = len(initial_angles[0])
    euler_dt = 0.001
    sample_rate = int(time_step/euler_dt)
    t = tf.linspace(0, total_time, int(total_time/euler_dt))
    x1 = np.zeros((n, int(total_time/euler_dt)))
    x2 = np.zeros((n, int(total_time/euler_dt)))
    v1 = np.zeros((n, int(total_time/euler_dt)))
    v2 = np.zeros((n, int(total_time/euler_dt)))
    x1[:,0] = np.radians(initial_angles[0])
    x2[:,0] = np.radians(initial_angles[1])
    v1[:,0] = np.radians(initial_angular_velocities[0])
    v2[:,0] = np.radians(initial_angular_velocities[1])
    def f(t, r):
        theta1 = r[0]
        theta2 = r[1]
        omega1 = r[2]
        omega2 = r[3]
        return np.array([omega, -g / l * np.sin(theta)])
    results = odeint(f, [np.radians(angle1), 0], t, tfirst=True)
    for i in range(1, int(total_time/euler_dt)):
        x1[:,i] = x1[:,i-1] + (v1[:,i-1]+np.random.normal(0, noise, (n,))) * euler_dt
        x2[:,i] = x2[:,i-1] + (v2[:,i-1]+np.random.normal(0, noise, (n,))) * euler_dt
        v[:,i] = v[:,i-1] + ((-g*(2*m1+m2)*np.sin(x1[:,i-1])-m2*g*np.sin(x1[:,i-1]-2*x2[:,i-1])-2*np.sin(x1[:,i-1]-x2[:,i-1])/())+np.random.normal(0, noise, (n,))) * euler_dt
    x = x[:,0::sample_rate]
    v = v[:,0::sample_rate]
    t = t[0::sample_rate]
    dx = (x[:,2:]-x[:,:-2])/(2*time_step)
    dv = (v[:,2:]-v[:,:-2])/(2*time_step)
    x = x[:,1:-1]
    v = v[:,1:-1]

    X = np.concatenate([x.reshape(1,-1).T,v.reshape(1,-1).T],1)
    Y = np.concatenate([dx.reshape(1,-1).T,dv.reshape(1,-1).T],1)
    return (X, Y)

def get_damped_pendulum_data(gamma, time_step, total_time, noise, initial_angles, initial_angular_velocities):
    g = l = 1
    w02 = g/l
    t = tf.linspace(0, total_time, int(total_time/time_step))
    def f(t, r):
        x = r[0]
        v = r[1]
        return np.array([v+np.random.normal(0,noise), -2*gamma*v-w02*np.sin(x)+np.random.normal(0,noise)])
    initial_conditions = np.array([initial_angles, initial_angular_velocities])
    X = np.apply_along_axis(lambda m: odeint(f, m, t, tfirst=True), 0, initial_conditions)
    Y = (X[2:,:,:] - X[:-2,:,:])/(2*time_step)
    X = X[1:-1,:,:]
    X = X.transpose(2,0,1).reshape(-1,2)
    Y = Y.transpose(2,0,1).reshape(-1,2)
    return (X, Y)

def get_grid_of_points(grid_range, grid_density):
    grid_xs = tf.linspace(-grid_range,grid_range,grid_density)
    grid_vs = tf.linspace(-grid_range,grid_range,grid_density)
    grid_xx, grid_vv = tf.meshgrid(grid_xs, grid_vs)
    grid_points = tf.stack([tf.reshape(grid_xx,[-1]), tf.reshape(grid_vv,[-1])], axis=1)
    return grid_points

def get_GPR_model(kernel, mean_function, data, iterations):
    X, Y = data
    m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,1,None],Y[:,0,None]],1)),(Y.shape[0]*2,1))), kernel=kernel, mean_function=mean_function)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=iterations))
    return m

def get_GPR_2Dmodel(kernel, mean_function, data, iterations):
    X, Y = data
    m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,2,None],Y[:,3,None],Y[:,0,None],Y[:,1,None]],1)),(Y.shape[0]*4,1))), kernel=kernel, mean_function=mean_function)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=iterations))
    return m

def evaluate_model(m, ground_truth, time_step):
    X, Y = ground_truth
    predicted = m.predict_f(X)[0]
    predicted_future = np.zeros(X.shape)
    predicted_future[0,:] = X[0,:]
    for i in range(1, X.shape[0]):
        pred = m.predict_f(to_default_float(predicted_future[i-1,:].reshape(1,2)))[0]
        predicted_future[i, 0] = predicted_future[i-1, 0] + pred[1]*time_step 
        predicted_future[i, 1] = predicted_future[i-1, 1] + pred[0]*time_step 
    MSE =  tf.reduce_mean(tf.math.square(predicted-tf.reshape(tf.transpose(tf.concat([Y[:,1,None],Y[:,0,None]],1)),(Y.shape[0]*2,1))))
    MSE_future = tf.reduce_mean(tf.math.square(predicted_future-X))
    return (MSE.numpy(), MSE_future.numpy(), predicted_future)

def evaluate_2Dmodel(m, ground_truth, time_step):
    X, Y = ground_truth
    predicted = m.predict_f(X)[0]
    predicted_future = np.zeros(X.shape)
    predicted_future[0,:] = X[0,:]
    for i in range(1, X.shape[0]):
        pred = m.predict_f(to_default_float(predicted_future[i-1,:].reshape(1,4)))[0]
        predicted_future[i, 0] = predicted_future[i-1, 0] + pred[2]*time_step 
        predicted_future[i, 1] = predicted_future[i-1, 1] + pred[3]*time_step 
        predicted_future[i, 2] = predicted_future[i-1, 2] + pred[0]*time_step 
        predicted_future[i, 3] = predicted_future[i-1, 3] + pred[1]*time_step 
    MSE =  tf.reduce_mean(tf.math.square(predicted-tf.reshape(tf.transpose(tf.concat([Y[:,2,None],Y[:,3,None],Y[:,0,None],Y[:,1,None]],1)),(Y.shape[0]*4,1))))
    MSE_future = tf.reduce_mean(tf.math.square(predicted_future-X))
    return (MSE.numpy(), MSE_future.numpy(), predicted_future)

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
