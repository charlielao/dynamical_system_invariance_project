# %%
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from gpflow.utilities import print_summary, positive
from termcolor import colored

# %%
t = tf.linspace(0, 100, 50)
x = tf.math.sin(t)
v = tf.math.cos(t)
plt.plot(t, x, "--")
plt.plot(t, v, "--")
# to sample the data randomly instead of regular spacing
#sampled_t = list(sorted(random.sample(list(t), 100)))
#sampled_x = 5*np.cos(sampled_t)
#sampled_v = 5*np.sin(sampled_t)
#plt.plot(sampled_t, sampled_x, 'x')
#plt.plot(sampled_t, sampled_v, 'x')
#plt.plot(x,v)
#plt.plot(sampled_x, sampled_v, "x")

# %%
X = tf.concat([x[:,None], v[:,None]], axis=-1)
X = tf.concat([X, 2*X], axis=0)
Y = (X[2:,:]-X[:-2, :])/(2) # to estimate acceleration and velocity by discrete differenation
X = X[1:-1, :]
plt.plot(X[:,1])
plt.plot(Y[:,0])
# %%
# regular GP
def fit_gp_posterior(X, Y, k, sigma2):
    Kff =k(X,X)  # K(X,X)
    Kffs = k(X, test_points)  # K(X,X*)
    Kfsfs = k(test_points)  # K(X*,X*)

    Y = tf.cast(Y, tf.float64)
    f_mean = tf.tensordot(tf.tensordot(tf.transpose(Kffs),tf.linalg.inv(Kff + sigma2*tf.eye(Kff.shape[0], dtype=tf.float64)),1), Y, 1)
    f_cov = Kfsfs-tf.tensordot(tf.tensordot(tf.transpose(Kffs), tf.linalg.inv(Kff+sigma2*tf.eye(Kff.shape[0],dtype=tf.float64)),1),Kffs,1)

    return f_mean, f_cov

# invariance GP
def fit_gp_posterior_invariance(X, Y, Ka, Kv, target, invar_grids, eval_grids, sigma2, mean):
    if target=="acc":
        k = Ka
        invar_grids_vector = invar_grids[:,1]
    else:
        k = Kv
        invar_grids_vector = invar_grids[:,0]

    K    = k(X,X)+tf.eye(X.shape[0], dtype=tf.float64)*sigma2  # K(X,X)
    Kg   = k(X,invar_grids)  # K(X,Xg)
    Ks   = k(X, eval_grids)  # K(X,X*)
    Kss  = k(eval_grids)  # K(X*,X*)
    Ksg  = k(eval_grids, invar_grids) #K(X*,Xg)
    Kgga = Ka(invar_grids) #K(Xg, Xg)
    Kggv = Kv(invar_grids) #K(Xg, Xg)
    Y = tf.cast(Y, tf.float64)
    
    invar_grids_matrix_x = tf.tensordot(invar_grids[:,0,None],invar_grids[None,:,0],1)
    invar_grids_matrix_x_dot = tf.tensordot(invar_grids[:,1,None],invar_grids[None,:,1],1)
    sum_invariance = tf.reduce_sum(invar_grids, -1, keepdims=True)

    Sigma11 = Kss
    Sigma12 = tf.concat([tf.transpose(Ks),tf.multiply(Ksg, invar_grids_vector)],-1)
    Sigma21 = tf.transpose(Sigma12)
    Sigma22 = tf.concat([tf.concat([K, tf.multiply(Kg, invar_grids_vector)], -1),tf.concat([tf.transpose(tf.multiply(Kg, invar_grids_vector)),tf.multiply(invar_grids_matrix_x, Kggv)+tf.multiply(invar_grids_matrix_x_dot, Kgga)],-1)],0)

    f_mean = mean*tf.ones((eval_grids.shape[0],1), dtype=tf.float64)+tf.tensordot(tf.tensordot(Sigma12,tf.linalg.inv(Sigma22),1),tf.concat([Y-mean*tf.ones((Y.shape[0],1),dtype=tf.float64),-mean*sum_invariance],0),1)
    f_cov = Sigma11-tf.tensordot(tf.tensordot(Sigma12, tf.linalg.inv(Sigma22),1),Sigma21,1)

    return f_mean, f_cov

# GP prior with invarinace
def fit_gp_prior_invariance(Ka, Kv, target, invar_grids, eval_grids, sigma2, mean):
    if target=="acc":
        k = Ka
        invar_grids_vector = invar_grids[:,1]
    else:
        k = Kv
        invar_grids_vector = invar_grids[:,0]

    Kss  = k(eval_grids)  # K(X*,X*)
    Ksg  = k(eval_grids, invar_grids) #K(X*,Xg)
    Kgga = Ka(invar_grids) #K(Xg, Xg)
    Kggv = Kv(invar_grids) #K(Xg, Xg)
    
    invar_grids_matrix_x = tf.tensordot(invar_grids[:,0,None],invar_grids[None,:,0],1)
    invar_grids_matrix_x_dot = tf.tensordot(invar_grids[:,1,None],invar_grids[None,:,1],1)
    sum_invariance = tf.reduce_sum(invar_grids, -1, keepdims=True)

    Sigma11 = Kss
    Sigma12 = tf.multiply(Ksg, invar_grids_vector)
    Sigma21 = tf.transpose(Sigma12)
    Sigma22 = tf.multiply(invar_grids_matrix_x, Kggv)+tf.multiply(invar_grids_matrix_x_dot, Kgga)

    f_mean = mean*tf.ones((eval_grids.shape[0],1), dtype=tf.float64)+tf.tensordot(tf.tensordot(Sigma12,tf.linalg.inv(Sigma22),1),-mean*sum_invariance,1)
    f_cov = Sigma11-tf.tensordot(tf.tensordot(Sigma12, tf.linalg.inv(Sigma22),1),Sigma21,1)

    return f_mean, f_cov
#regular GP lml
def reg_lml(X, Y, k, sigma2):
    Kff = k(X)  # K(X,X)
    A = Kff + sigma2 * tf.eye(Kff.shape[0], dtype=tf.float64)
    A_inverse = tf.linalg.inv(A)
    lml = -0.5 * tf.tensordot((tf.tensordot(tf.transpose(Y), A_inverse,1)),Y,1) -0.5*tf.math.log(tf.linalg.det(A)) -0.5* tf.cast(X.shape[0] * tf.math.log(2 * np.pi), tf.float64)
    return lml[0, 0]

#invariance GP lml
def inv_lml(X, Y, Ka, Kv, target, invar_grids, sigma2):
    if target=="acc":
        k = Ka
        invar_grids_vector = invar_grids[:,1]
    else:
        k = Kv
        invar_grids_vector = invar_grids[:,0]

    K    = k(X,X) + tf.eye(X.shape[0], dtype=tf.float64)*sigma2  # K(X,X)
    Kg   = k(X,invar_grids)  # K(X,Xg)
    Kgga = Ka(invar_grids) #K(Xg, Xg)
    Kggv = Kv(invar_grids) #K(Xg, Xg)
    Y = tf.cast(Y, tf.float64)
    
    invar_grids_matrix_x = tf.tensordot(invar_grids[:,0,None],invar_grids[None,:,0],1)
    invar_grids_matrix_x_dot = tf.tensordot(invar_grids[:,1,None],invar_grids[None,:,1],1)
    A = K-tf.tensordot(tf.tensordot(tf.multiply(Kg, invar_grids_vector),tf.linalg.inv(tf.multiply(invar_grids_matrix_x, Kggv)+tf.multiply(invar_grids_matrix_x_dot, Kgga)),1), tf.transpose(tf.multiply(Kg, invar_grids_vector)),1)

    A_inverse = tf.linalg.inv(A)
    lml = -0.5 * tf.tensordot((tf.tensordot(tf.transpose(Y), A_inverse,1)),Y,1) -0.5*tf.math.log(tf.linalg.det(A)) -0.5* tf.cast(X.shape[0] * tf.math.log(2 * np.pi), tf.float64)

    return lml[0, 0]

# plotting
def plotting(pred, var, inv, save, name):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(test_xx, test_vv, tf.reshape(pred, test_xx.shape), cmap="viridis",linewidth=0, antialiased=False, alpha=0.5)
    surf = ax.plot_surface(test_xx, test_vv, tf.reshape(pred, test_xx.shape)+1.96*tf.math.sqrt(tf.reshape(tf.linalg.tensor_diag_part(var), test_xx.shape)), color="grey",linewidth=0, antialiased=False, alpha=0.3)
    surf = ax.plot_surface(test_xx, test_vv, tf.reshape(pred, test_xx.shape)-1.96*tf.math.sqrt(tf.reshape(tf.linalg.tensor_diag_part(var), test_xx.shape)), color="grey",linewidth=0, antialiased=False, alpha=0.3)
    #surf = ax.plot_surface(test_xx, test_vv, -test_xx, color="grey",linewidth=0, antialiased=False, alpha=0.1)
    #surf = ax.scatter(test_xx, test_vv, tf.reshape(pred, test_xx.shape), cmap="viridis",linewidth=0, antialiased=False, alpha=0.3)
    ax.scatter(X[:,0], X[:,1],Y[:,1,None], color="black", marker="o", s=3) #data
    if inv=="inv":
        inv_p = ax.scatter(invariance_points[:,0], invariance_points[:,1], tf.reshape(pred_inv, invariance_points[:,0].shape), color="red", marker="o", s=3, alpha=0.2)
    ax.view_init(10,-65)
    #cax = fig.add_axes([0.9, .3, 0.02, 0.6])
    #fig.colorbar(surf, cax=cax)
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_zlabel("acceleration")
    #ax.annotate("log marginal likelihood: {0:0.2f}".format(m.log_marginal_likelihood()), xy=(0.5, 0.9), xycoords='axes fraction')
    if save:
        plt.savefig(name+"_3D.pdf")
    plt.figure(figsize=(5, 3))
    contours = plt.contourf(test_xx, test_vv, tf.reshape(pred,(test_xx.shape)), levels=100, cmap="viridis", alpha=0.3)
    plt.colorbar(contours)
    contours = plt.scatter(X[:,0],X[:,1],c=Y[:,1,None],cmap="viridis", alpha=0.2)
    if inv=="inv":
        contours =  plt.scatter(invariance_points[:,0], invariance_points[:,1],s=1, color="red", marker="o", alpha=0.5)

    plt.xlim((-test_range, test_range))
    plt.ylim((-test_range, test_range))
    plt.xlabel("position")
    plt.ylabel("velocity")
    if save:
        plt.savefig(name+"_contour.pdf")
# %%
# range of invariance we are conditioning on
invariance_range = 5
invariance_density = 10
invariance_xs = tf.linspace(-invariance_range,invariance_range,invariance_density)
invariance_vs = tf.linspace(-invariance_range,invariance_range,invariance_density)
invariance_xx, invariance_vv = tf.meshgrid(invariance_xs, invariance_vs)
invariance_points = tf.stack([tf.reshape(invariance_xx,[-1]), tf.reshape(invariance_vv,[-1])], axis=1)

# range we are evaluating the test points on
test_range = 5 
test_density = 40
test_xs = tf.linspace(-test_range,test_range,test_density)
test_vs = tf.linspace(-test_range,test_range,test_density)
test_xx, test_vv = tf.meshgrid(test_xs, test_vs)
test_points = tf.stack([tf.reshape(test_xx,[-1]), tf.reshape(test_vv,[-1])], axis=1)
# %%
# kernel used
ka = gpflow.kernels.RBF(variance=1, lengthscales=[1,1])
kv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1])
# %%
# Invariance GP prediction and variance
mean_ = 0
sigma2_= 0.01
pred, var = fit_gp_posterior_invariance(X, Y[:,1,None], ka, kv, "acc", invariance_points, test_points, sigma2_, mean_)

# invarinace points prediction 
pred_inv, _ = fit_gp_posterior_invariance(X, Y[:,1,None], ka, kv, "acc", invariance_points, invariance_points, sigma2_ ,mean_)

#invariance prior
pred_prior, var_prior = fit_gp_prior_invariance(ka, kv, "acc", invariance_points, test_points, sigma2_, mean_)


# regular GP prediction
reg_pred, reg_var = fit_gp_posterior(X, Y[:,1, None], ka, sigma2_)

# %%
print(reg_lml(X, Y[:,1,None], ka, sigma2_))
print(inv_lml(X, Y[:,1,None], ka, kv, "acc", invariance_points, sigma2_))
# %%
plotting(pred_prior, var_prior, "inv", 0 , "prior_invariance")
plotting(pred, var+sigma2_, "inv", 0, "posterior_invariance")
plotting(reg_pred, reg_var+sigma2_, "", 0, "regular")

# %%
m = gpflow.models.GPR(data=(X, Y[:, 1,None]), kernel=ka, mean_function=None)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
pred, var = m.predict_f(test_points)
plotting(pred, reg_var, "nv", 0, "whatever")

# %%
# hyperparameter tuning
invariance_density_tuning = [10, 20, 30]
length_scales_tuning = [0.1, 1, 5]
likelihood_variance_tuning = [0.01, 0.1, 1]

def tuning(invariance_density_range, length_scales_range, likelihood_variance_range):
    max_inv_lml = -float(np.inf)
    for invariance_density_t in invariance_density_range:
        for length_scales_t in length_scales_range:
            for likelihood_variance_t in likelihood_variance_range:
                invariance_range = 5
                invariance_density = invariance_density_t
                invariance_xs = tf.linspace(-invariance_range,invariance_range,invariance_density)
                invariance_vs = tf.linspace(-invariance_range,invariance_range,invariance_density)
                invariance_xx, invariance_vv = tf.meshgrid(invariance_xs, invariance_vs)
                invariance_points = tf.stack([tf.reshape(invariance_xx,[-1]), tf.reshape(invariance_vv,[-1])], axis=1)
                ka = gpflow.kernels.RBF(variance=1, lengthscales=[length_scales_t,length_scales_t])
                kv = gpflow.kernels.RBF(variance=1, lengthscales=[length_scales_t,length_scales_t])
                mean_ = 0
                sigma2_= likelihood_variance_t
                pred, var = fit_gp_posterior_invariance(X, Y[:,1,None], ka, kv, "acc", invariance_points, test_points, sigma2_, mean_)

                pred_inv, _ = fit_gp_posterior_invariance(X, Y[:,1,None], ka, kv, "acc", invariance_points, invariance_points, sigma2_ ,mean_)

                reg_pred, reg_var = fit_gp_posterior(X, Y[:,1, None], ka, sigma2_)
                regular_lml = reg_lml(X, Y[:,1,None], ka, sigma2_).numpy()
                invariance_lml = inv_lml(X, Y[:,1,None], ka, kv, "acc", invariance_points, sigma2_).numpy()
                if invariance_lml > max_inv_lml:
                    max_inv_lml = invariance_lml
                    colour="yellow"
                else:
                    colour="white"

                print(colored("invariance_density: %s, lengthscales: %s, likelihood variance: %s\n"%(invariance_density_t, length_scales_t, likelihood_variance_t),colour))
                print(colored("regular likelihood: %s\n"%(regular_lml), colour))
                print(colored("invariance likelihood: %s\n"%(invariance_lml), colour))
    return max_inv_lml
max_t = tuning(invariance_density_tuning, length_scales_tuning, likelihood_variance_tuning)
# %%


'''
# Try to write a kernel to condition on 
class SHO_Energy_Invariance(gpflow.kernels.Kernel):
    def __init__(self, invariance_range):
        super().__init__(active_dims=[0])
        self.invar_density = tf.Variable(20)
        self.Ka = gpflow.kernels.RBF(variance=1, lengthscales=[1,1])
        self.Kv = gpflow.kernels.RBF(variance=1, lengthscales=[1,1])
        invariance_xs = tf.linspace(-invariance_range,invariance_range,self.invar_density)
        invariance_vs = tf.linspace(-invariance_range,invariance_range,self.invar_density)
        invariance_xx, invariance_vv = tf.meshgrid(invariance_xs, invariance_vs)
        self.invar_grids = tf.stack([tf.reshape(invariance_xx,[-1]), tf.reshape(invariance_vv,[-1])], axis=1)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
#        if acc:
        k = self.Ka
        invar_grids_vector = self.invar_grids[:,1]
 #       else:
  #          k = self.Kv
   #         invar_grids_vector = self.invar_grids[:,0]

        K11  = k(X, X) #K(X, X) 
        K22  = k(X2, X2) #K(X2, X2)
        K12  = k(X,X2) # K(X,X2)
        K21 = tf.transpose(K12)
        K1g  = k(X, self.invar_grids) #K(X,Xg)
        K2g =  k(X2, self.invar_grids) #K(X2,Xg)
        Kgga = self.Ka(self.invar_grids) #K(Xg, Xg)
        Kggv = self.Kv(self.invar_grids) #K(Xg, Xg)
        
        invar_grids_matrix_x = tf.tensordot(self.invar_grids[:,0,None],self.invar_grids[None,:,0],1)
        invar_grids_matrix_x_dot = tf.tensordot(self.invar_grids[:,1,None],self.invar_grids[None,:,1],1)

        Sigma11 = tf.concat([tf.concat([K11,K12],1),tf.concat([K21, K22],1)],0)
        Sigma12 = tf.concat([tf.multiply(K1g, invar_grids_vector),tf.multiply(K2g, invar_grids_vector)],0)
        Sigma21 = tf.transpose(Sigma12)
        Sigma22 = tf.multiply(invar_grids_matrix_x, Kggv)+tf.multiply(invar_grids_matrix_x_dot, Kgga)

        return Sigma11-tf.tensordot(tf.tensordot(Sigma12, tf.linalg.inv(Sigma22),1),Sigma21,1)

    def K_diag(self, X):
       return tf.linalg.tensor_diag_part(self.K(X))


energy_invariance = SHO_Energy_Invariance(5)
energy_invariance(X,X)

'''
# %%
