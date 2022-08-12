import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable

def degree_of_freedom(kernel, grid_range, grid_density, likelihood):
    test_points = get_grid_of_points_1D(grid_range, grid_density)
    K = kernel(test_points)
    return tf.linalg.trace(tf.tensordot(K, tf.linalg.inv(K+likelihood*tf.eye(K.shape[0], dtype=tf.float64)), 1)).numpy()

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

def get_SHM_data_2D(time_step, total_time, noise, initial_positions_1,initial_positions_2, initial_velocities_1,initial_velocities_2):
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

def get_grid_of_points_1D(grid_range, grid_density):
    grid_xs = tf.linspace(-grid_range,grid_range,grid_density)
    grid_vs = tf.linspace(-grid_range,grid_range,grid_density)
    grid_xx, grid_vv = tf.meshgrid(grid_xs, grid_vs)
    grid_points = tf.stack([tf.reshape(grid_xx,[-1]), tf.reshape(grid_vv,[-1])], axis=1)
    return grid_points

def get_grid_of_points_2D(grid_range, grid_density):
    grid_x1s = tf.linspace(-grid_range,grid_range,grid_density)
    grid_x2s = tf.linspace(-grid_range,grid_range,grid_density)
    grid_v1s = tf.linspace(-grid_range,grid_range,grid_density)
    grid_v2s = tf.linspace(-grid_range,grid_range,grid_density)
    grid_xx1, grid_xx2, grid_vv1, grid_vv2 = tf.meshgrid(grid_x1s, grid_x2s, grid_v1s, grid_v2s)
    grid_points = tf.stack([tf.reshape(grid_xx1,[-1]),tf.reshape(grid_xx2,[-1]),tf.reshape(grid_vv1,[-1]), tf.reshape(grid_vv2,[-1])], axis=1)
    return grid_points

def callback(step, variables, values):
    print(step, end='\r')

def get_GPR_model(kernel, mean_function, data, iterations):

#    def callback(step, variables, values):
#        if step%10==0:
#            stored_f.append(variables[0].numpy())
#            stored_g.append(variables[1].numpy())
#            stored_coeff.append(variables)
#    stored_f = []; stored_g =[] 
#    stored_coeff = []

    X, Y = data
    m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,1,None],Y[:,0,None]],1)),(Y.shape[0]*2,1))), kernel=kernel, mean_function=mean_function)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=iterations), step_callback=callback)
    return m#, stored_f, stored_g, stored_coeff

def get_GPR_model_2D(kernel, mean_function, data, iterations, old_model=None):
    X, Y = data
    m = gpflow.models.GPR(data=(X, tf.reshape(tf.transpose(tf.concat([Y[:,2,None],Y[:,3,None],Y[:,0,None],Y[:,1,None]],1)),(Y.shape[0]*4,1))), kernel=kernel, mean_function=mean_function)
    opt = gpflow.optimizers.Scipy()
    if old_model:
        kernel.Ka1.lengthscales = gpflow.Parameter(old_model.kernel.Ka1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
        kernel.Ka2.lengthscales = gpflow.Parameter(old_model.kernel.Ka2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
        kernel.Kv1.lengthscales = gpflow.Parameter(old_model.kernel.Kv1.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
        kernel.Kv2.lengthscales = gpflow.Parameter(old_model.kernel.Kv2.lengthscales.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
        kernel.Ka1.variance = gpflow.Parameter(old_model.kernel.Ka1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
        kernel.Ka2.variance = gpflow.Parameter(old_model.kernel.Ka2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
        kernel.Kv1.variance = gpflow.Parameter(old_model.kernel.Kv1.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
        kernel.Kv2.variance = gpflow.Parameter(old_model.kernel.Kv2.variance.numpy(), transform=tfp.bijectors.Sigmoid(to_default_float(1e-3), to_default_float(5.))) 
        kernel.local_invar_grid = old_model.kernel.local_invar_grid
        m.likelihood.variance = gpflow.Parameter(old_model.likelihood.variance, transform=positive(lower=1e-6-1e-12))

#   log_dir_scipy = f"{log_dir}/scipy"
#    model_task = ModelToTensorBoard(log_dir_scipy, model)
#    lml_task = ScalarToTensorBoard(log_dir_scipy, lambda: model.training_loss(), "training_objective")

#    monitor = Monitor(MonitorTaskGroup([model_task, lml_task], period=1), MonitorTaskGroup(image_task, period=5))
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables,  options=dict(maxiter=iterations), step_callback=callback)
    return m

'''
class GPR_with_sparse(gpflow.models.GPR):
    def __init__(self, reg, **kwargs):
        super().__init__(**kwargs)
        self.reg = reg
    def maximum_log_likelihood_objective(self):
        return self.log_marginal_likelihood()-self.reg*(tf.reduce_sum(tf.abs(self.kernel.poly)))
'''

def get_GPR_model_GD_2D(model, iterations, lr, manager):
    opt = tf.optimizers.Adam(learning_rate=lr)

    @tf.function
    def optimization_step(m):
        opt.minimize(m.training_loss, m.trainable_variables)

    for j in range(iterations):
        optimization_step(model)
        lml = model.log_marginal_likelihood().numpy()
        print(round(lml)," ", j, end="\r")#, end="\r")#,np.array2string(tf.concat([m.kernel.f1_poly,m.kernel.f2_poly,m.kernel.g1_poly,m.kernel.g2_poly],1).numpy()))
        if j%1000==0:
            manager.save()
#    m.kernel.poly.assign(tf.map_fn(lambda x: tf.where(abs(x)<1e-5, 0 , x), m.kernel.poly.numpy()))
#    for j in range(int(iterations/1000)):
#        optimization_step(model)
#        lml = model.log_marginal_likelihood().numpy()
#        print(round(lml)," ", j, end="\r")#, end="\r")#,np.array2string(tf.concat([m.kernel.f1_poly,m.kernel.f2_poly,m.kernel.g1_poly,m.kernel.g2_poly],1).numpy()))
    return model

def evaluate_model_future(m, test_starting, dynamics, time_setting, energy):
    test_starting_position, test_starting_velocity = test_starting
    total_time, time_step = time_setting
    likelihood = m.likelihood.variance.numpy()
    X = np.zeros((int(total_time/time_step),2))
    X[0,0] = test_starting_position
    X[0,1] = test_starting_velocity

    predicted_future = np.zeros(X.shape)
    predicted_future_variance_top = np.zeros(X.shape)
    predicted_future_variance_bottom = np.zeros(X.shape)
    predicted_future[0,:] = X[0,:]

    X[1,0] = X[0,0] + X[0,1]*time_step
    X[1,1] = X[0,1] + dynamics(X[None,0,:])*time_step

    pred, var = m.predict_f(to_default_float(predicted_future[0,:].reshape(1,2)))

    predicted_future[1, 0] = predicted_future[0, 0] + pred[1]*time_step 
    predicted_future_variance_top[1, 0] = predicted_future[0, 0] + (pred[1]+1.96*np.sqrt(var[1]+likelihood))*time_step
    predicted_future_variance_bottom[1, 0] = predicted_future[0, 0] - (pred[1]+1.96*np.sqrt(var[1]+likelihood))*time_step

    predicted_future[1, 1] = predicted_future[0, 1] + pred[0]*time_step 
    predicted_future_variance_top[1, 1] = predicted_future[0, 1] + (pred[0]+1.96*np.sqrt(var[0]+likelihood))*time_step
    predicted_future_variance_bottom[1, 1] = predicted_future[0, 1] - (pred[0]+1.96*np.sqrt(var[0]+likelihood))*time_step

    for i in range(2, X.shape[0]):
        pred, var = m.predict_f(to_default_float(predicted_future[i-1,:].reshape(1,2)))

        predicted_future[i, 0] = predicted_future[i-2, 0] + pred[1]*2*time_step 
        X[i,0] =X[i-2,0] + X[i-1,1]*2*time_step 
        predicted_future_variance_top[i, 0] = predicted_future[i-2, 0] + (pred[1]+1.96*np.sqrt(var[1]+likelihood))*2*time_step
        predicted_future_variance_bottom[i, 0] = predicted_future[i-2, 0] - (pred[1]+1.96*np.sqrt(var[1]+likelihood))*2*time_step

        predicted_future[i, 1] = predicted_future[i-2, 1] + pred[0]*2*time_step 
        X[i,1] =X[i-2,1] + dynamics(X[None,i-1,:])*2*time_step 
        predicted_future_variance_top[i, 1] = predicted_future[i-2, 1] + (pred[0]+1.96*np.sqrt(var[0]+likelihood))*2*time_step
        predicted_future_variance_bottom[i, 1] = predicted_future[i-2, 1] - (pred[0]+1.96*np.sqrt(var[0]+likelihood))*2*time_step
    MSE_future = tf.reduce_mean(tf.math.square(predicted_future-X))
    true_energy = energy(X)
    predicted_energy = energy(predicted_future)
    return (MSE_future.numpy(), predicted_future, predicted_future_variance_top, predicted_future_variance_bottom, X, true_energy, predicted_energy)

def evaluate_model_grid(m, grids, dynamics):
    grid_range, grid_density = grids
    X = get_grid_of_points_1D(grid_range, grid_density)
    predicted = m.predict_f(X)[0]
    predicted = tf.transpose(tf.reshape(predicted, (2, int(predicted.shape[0]/2))))
    Y = dynamics(X) #acceleration
    MSE =  tf.reduce_mean(tf.math.square(predicted-tf.concat([Y[:,None],X[:,1,None]],1)))
    return MSE.numpy()

def evaluate_model_future_2D(m, test_starting, dynamics, time_setting, scalers, energy):
    test_starting_position1, test_starting_position2, test_starting_velocity1, test_starting_velocity2 = test_starting
    dynamics1, dynamics2 = dynamics
    total_time, time_step = time_setting
    scalerX, scalerY = scalers
    likelihood = m.likelihood.variance.numpy()

    X = np.zeros((int(total_time/time_step),4))
    X[0,0] = test_starting_position1
    X[0,1] = test_starting_position2
    X[0,2] = test_starting_velocity1
    X[0,3] = test_starting_velocity2

    predicted_future = np.zeros(X.shape)
    predicted_future_variance_top = np.zeros(X.shape)
    predicted_future_variance_bottom = np.zeros(X.shape)
    predicted_future[0,:] = X[0,:]

    X[1,0] = X[0,0] + X[0,2]*time_step
    X[1,1] = X[0,1] + X[0,3]*time_step
    X[1,2] = X[0,2] + dynamics1(X[None,0,:])*time_step
    X[1,3] = X[0,3] + dynamics2(X[None,0,:])*time_step

    pred, var = m.predict_f(scalerX.transform(to_default_float(predicted_future[0,:].reshape(1,4))))
    pred = tf.roll(tf.transpose(pred), shift=-2, axis=1)
    var =  tf.roll(tf.transpose(var), shift=-2, axis=1)
    var = scalerY.inverse_transform(pred + 1.96*np.sqrt(var+likelihood*np.ones(var.shape)))
    pred = scalerY.inverse_transform(pred)

    predicted_future[1, :] = predicted_future[0, :] + pred*time_step 
    predicted_future_variance_top[1, :] = predicted_future[0, :] + var*time_step
    predicted_future_variance_bottom[1, :] = predicted_future[0, :] - var*time_step

    for i in range(2, X.shape[0]):
        if i==2:
            start = time.time()
        if i==8:
            average_time_per_step = (time.time()-start)/6
        if i>8:
            print("%s seconds remaining"%round(average_time_per_step*(X.shape[0]-i)), end="\r")
        pred, var = m.predict_f(scalerX.transform(to_default_float(predicted_future[i-1,:].reshape(1,4))))
        pred = tf.roll(tf.transpose(pred), shift=-2, axis=1)
        var =  tf.roll(tf.transpose(var), shift=-2, axis=1)
        var = scalerY.inverse_transform(pred + 1.96*np.sqrt(var+likelihood*np.ones(var.shape)))
        pred = scalerY.inverse_transform(pred)

        predicted_future[i, :] = predicted_future[i-2, :] + pred*2*time_step 
        predicted_future_variance_top[i, :] = predicted_future[i-2, :] + var*2*time_step
        predicted_future_variance_bottom[i, :] = predicted_future[i-2, :] - var*2*time_step
        X[i,0] = X[i-2,0] + X[i-1,2]*2*time_step
        X[i,1] = X[i-2,1] + X[i-1,3]*2*time_step
        X[i,2] = X[i-2,2] + dynamics1(X[None,i-1,:])*2*time_step
        X[i,3] = X[i-2,3] + dynamics2(X[None,i-1,:])*2*time_step

    MSE_future = tf.reduce_mean(tf.math.square(predicted_future-X))
    true_energy = energy(X)
    predicted_energy = energy(predicted_future)
    return (MSE_future.numpy(), predicted_future, predicted_future_variance_top, predicted_future_variance_bottom, X, true_energy, predicted_energy)

def evaluate_model_grid_2D(m, grids, dynamics, scalers):
    grid_range, grid_density = grids
    dynamics1, dynamics2 = dynamics
    scalerX, scalerY = scalers
    X = get_grid_of_points_2D(grid_range, grid_density)
    Y1 = dynamics1(X) #acceleration1
    Y2 = dynamics2(X) #acceleration2
    predicted = m.predict_f(scalerX.transform(X))[0]
    predicted = tf.transpose(tf.reshape(predicted, (4, int(predicted.shape[0]/4))))
    predicted = tf.roll(predicted, shift=-2, axis=1)
    predicted = scalerY.inverse_transform(predicted)
    MSE =  tf.reduce_mean(tf.math.square(predicted-tf.concat([X[:,2,None], X[:,3,None],Y1[:,None], Y2[:,None]],1)))
    return MSE.numpy()

def SHM_dynamics(X):
    return -X[:,0]

def SHM_dynamics1_2D(X):
    return -X[:,0]

def SHM_dynamics2_2D(X):
    return -X[:,1]

def pendulum_dynamics(X):
    return -np.sin(X[:,0])

def double_pendulum_dynamics1(X):
    x1 = X[:,0]; x2 = X[:,1]; v1 = X[:,2]; v2 = X[:,3]
    numerator = -3*np.sin(x1)-np.sin(x1-2*x2)-2*np.sin(x1-x2)*(v2**2+v1**2*np.cos(x1-x2))
    denominator = 3-np.cos(2*x1-2*x2)
    return numerator/denominator

def double_pendulum_dynamics2(X):
    x1 = X[:,0]; x2 = X[:,1]; v1 = X[:,2]; v2 = X[:,3]
    numerator = 2*np.sin(x1-x2)*(2*v1**2+2*np.cos(x1)+v2**2*np.cos(x1-x2))
    denominator = 3-np.cos(2*x1-2*x2)
    return numerator/denominator

def damped_SHM_dynamics(X, gamma=0.1):
    return -X[:,0]-2*gamma*X[:,1]

def damped_pendulum_dynamics(X, gamma=0.1):
    return -np.sin(X[:,0])-2*gamma*X[:,1]


