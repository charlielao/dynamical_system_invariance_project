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
    t = tf.linspace(0., total_time, int(total_time/time_step))
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
    t = tf.linspace(0., total_time, int(total_time/time_step))
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
    t = tf.linspace(0., total_time, int(total_time/time_step))
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
    t = tf.linspace(0., total_time, int(total_time/time_step))
    def f(t, r):
        x = r[0]
        v = r[1]
        return np.array([v+np.random.normal(0,noise), -w02*np.sin(x)+np.random.normal(0,noise)])
    initial_conditions = np.radians(np.array([initial_angles, initial_angular_velocities]))
    X = np.apply_along_axis(lambda m: odeint(f, m, t, tfirst=True), 0, initial_conditions)
    Y = (X[2:,:,:] - X[:-2,:,:])/(2*time_step)
    X = X[1:-1,:,:]
    X = X.transpose(2,0,1).reshape(-1,2)
    Y = Y.transpose(2,0,1).reshape(-1,2)
    return (X, Y)

def get_double_pendulum_data(time_step, total_time, noise, initial_angles_1, initial_angles_2, initial_angular_velocities_1, initial_angular_velocities_2):
    m1 = m2 = l1 = l2 = g = 1
    t = tf.linspace(0., total_time, int(total_time/time_step))
    def f(t, r):
        x1 = r[0]
        x2 = r[1]
        v1 = r[2]
        v2 = r[3]
        dv1 = (-g*(2*m1+m2)*np.sin(x1)-m2*g*np.sin(x1-2*x2)-2*np.sin(x1-x2)*m2*(v2**2*l2+v1**2*l1*np.cos(x1-x2)))/(l1*(2*m1+m2-m2*np.cos(2*x1-2*x2)))
        dv2 = (2*np.sin(x1-x2)*(v1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(x1)+v2**2*l2*m2*np.cos(x1-x2)))/(l2*(2*m1+m2-m2*np.cos(2*x1-2*x2)))
        return np.array([v1+np.random.normal(0,noise), v2+np.random.normal(0,noise),dv1+np.random.normal(0,noise),dv2+np.random.normal(0,noise)])
    initial_conditions = np.radians(np.array([initial_angles_1, initial_angles_2, initial_angular_velocities_1, initial_angular_velocities_2]))
    X = np.apply_along_axis(lambda m: odeint(f, m, t, tfirst=True), 0, initial_conditions)
    Y = (X[2:,:,:] - X[:-2,:,:])/(2*time_step)
    X = X[1:-1,:,:]
    X = X.transpose(2,0,1).reshape(-1,4)
    Y = Y.transpose(2,0,1).reshape(-1,4)
    return (X, Y)

def get_damped_pendulum_data(gamma, time_step, total_time, noise, initial_angles, initial_angular_velocities):
    g = l = 1
    w02 = g/l
    t = tf.linspace(0., total_time, int(total_time/time_step))
    def f(t, r):
        x = r[0]
        v = r[1]
        return np.array([v+np.random.normal(0,noise), -2*gamma*v-w02*np.sin(x)+np.random.normal(0,noise)])
    initial_conditions = np.radians(np.array([initial_angles, initial_angular_velocities]))
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

def callback(step, variables, values):
    print(step, end='\r')

    

def get_GPR_model(kernel, mean_function, data, iterations):
    X, Y = data
    m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,1,None],Y[:,0,None]],1)),(Y.shape[0]*2,1))), kernel=kernel, mean_function=mean_function)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=iterations), step_callback=callback)
    return m

def get_GPR_2Dmodel(kernel, mean_function, data, optimiser, iterations, lr):
    X, Y = data
    m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,2,None],Y[:,3,None],Y[:,0,None],Y[:,1,None]],1)),(Y.shape[0]*4,1))), kernel=kernel, mean_function=mean_function)
    if optimiser=="scipy":
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=iterations), step_callback=callback)
        return m
    else:
        opt = tf.optimizers.Adam(learning_rate=lr)
        @tf.function
        def optimization_step():
            opt.minimize(m.training_loss, m.trainable_variables)
        best = m.log_marginal_likelihood().numpy()
        for _ in range(iterations):
            optimization_step()
            lml = m.log_marginal_likelihood().numpy()
            if lml > best:
                if lml-best<1e-6:
                    break
                best = lml
                best_param = m.trainable_variables
            print(round(lml)," ", _,end='\r')
        return m, best_param

class GPR_with_sparse(gpflow.models.GPR):
    def __init__(self, reg, **kwargs):
        super().__init__(**kwargs)
        self.reg = reg
    def maximum_log_likelihood_objective(self):
        return self.log_marginal_likelihood()-self.reg*(tf.reduce_sum(tf.abs(self.kernel.f1_poly))+tf.reduce_sum(tf.abs(self.kernel.f2_poly))+tf.reduce_sum(tf.abs(self.kernel.g1_poly))+tf.reduce_sum(tf.abs(self.kernel.g2_poly)))

def get_GPR_2Dmodel_sparse(kernel, mean_function, data, optimiser, iterations, lr, reg, drop_rate):
    X, Y = data
    
    m = GPR_with_sparse(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,2,None],Y[:,3,None],Y[:,0,None],Y[:,1,None]],1)),(Y.shape[0]*4,1))), kernel=kernel, mean_function=mean_function, reg=reg)
    if optimiser=="scipy":
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=iterations), step_callback=callback)
        return m
    else:
        opt = tf.optimizers.Adam(learning_rate=lr)
        @tf.function
        def optimization_step():
            opt.minimize(m.training_loss, m.trainable_variables)
        best = m.log_marginal_likelihood().numpy()
        for _ in range(iterations):
            optimization_step()
            lml = m.log_marginal_likelihood().numpy()
            if lml > best:
                if lml-best<1e-8:
                    break
                best = lml
                best_param = m.trainable_variables
            for i,_ in enumerate(m.kernel.f1_poly):
                if np.random.uniform()<drop_rate:
                    drop = m.kernel.f1_poly.numpy()
                    drop[i] = 0
                    m.kernel.f1_poly.assign(drop)
            for i,_ in enumerate(m.kernel.f2_poly):
                if np.random.uniform()<drop_rate:
                    drop = m.kernel.f2_poly.numpy()
                    drop[i] = 0
                    m.kernel.f2_poly.assign(drop)
            for i,_ in enumerate(m.kernel.g1_poly):
                if np.random.uniform()<drop_rate:
                    drop = m.kernel.g1_poly.numpy()
                    drop[i] = 0
                    m.kernel.g1_poly.assign(drop)
            for i,_ in enumerate(m.kernel.g2_poly):
                if np.random.uniform()<drop_rate:
                    drop = m.kernel.g2_poly.numpy()
                    drop[i] = 0
                    m.kernel.g2_poly.assign(drop)
            try:
                print(round(lml)," ", _)
            except ValueError:
                print("bad coefficients")
            print(tf.concat([m.kernel.f1_poly,m.kernel.f2_poly,m.kernel.g1_poly,m.kernel.g2_poly],1).numpy())
        return m, best_param

def evaluate_model(m, ground_truth, time_step):
    X, Y = ground_truth
    predicted = m.predict_f(X)[0]
    MSE =  tf.reduce_mean(tf.math.square(predicted-tf.reshape(tf.transpose(tf.concat([Y[:,1,None],Y[:,0,None]],1)),(Y.shape[0]*2,1))))

    predicted_future = np.zeros(X.shape)
    predicted_future_variance = np.zeros(X.shape)
    predicted_future[0,:] = X[0,:]
    for i in range(1, X.shape[0]):
        pred, var = m.predict_f(to_default_float(predicted_future[i-1,:].reshape(1,2)))
        predicted_future[i, 0] = predicted_future[i-1, 0] + pred[1]*time_step 
        predicted_future_variance[i, 0] = var[1]
        predicted_future[i, 1] = predicted_future[i-1, 1] + pred[0]*time_step 
        predicted_future_variance[i, 1] = var[0]
    MSE_future = tf.reduce_mean(tf.math.square(predicted_future-X))
    return (MSE.numpy(), MSE_future.numpy(), predicted_future, predicted_future_variance)

def evaluate_2Dmodel(m, ground_truth, time_step):
    X, Y = ground_truth
    predicted = m.predict_f(X)[0]
    MSE =  tf.reduce_mean(tf.math.square(predicted-tf.reshape(tf.transpose(tf.concat([Y[:,2,None],Y[:,3,None],Y[:,0,None],Y[:,1,None]],1)),(Y.shape[0]*4,1))))
    predicted_future = np.zeros(X.shape)
    predicted_future_variance = np.zeros(X.shape)
    predicted_future[0,:] = X[0,:]
    for i in range(1, X.shape[0]):
        pred, var = m.predict_f(to_default_float(predicted_future[i-1,:].reshape(1,4)))
        predicted_future[i, 0] = predicted_future[i-1, 0] + pred[2]*time_step 
        predicted_future_variance[i, 0] = var[2]
        predicted_future[i, 1] = predicted_future[i-1, 1] + pred[3]*time_step 
        predicted_future_variance[i, 1] = var[3]
        predicted_future[i, 2] = predicted_future[i-1, 2] + pred[0]*time_step 
        predicted_future_variance[i, 2] = var[0]
        predicted_future[i, 3] = predicted_future[i-1, 3] + pred[1]*time_step 
        predicted_future_variance[i, 3] = var[1]
    MSE_future = tf.reduce_mean(tf.math.square(predicted_future-X))
    return (MSE.numpy(), MSE_future.numpy(), predicted_future, predicted_future_variance)

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
