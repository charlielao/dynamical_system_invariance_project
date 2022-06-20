import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
import invariance_functions as inv

test_grids = inv.get_test_points()
print("SHM")
for gamma in [0.01, 0.05, 0.1]:
    print("current damping: %s" %gamma)
    data = inv.get_damped_SHM_data(gamma)
    for fixed in [True, False]:
        if fixed:
            fixed_mean = "fixed mean"
        else:
            fixed_mean = "damping mean"
        print(fixed_mean)
        for jitter in [1e-5, 1e-6]:
            print("current jitter %s" %jitter)
            print("no invariance has log marginal likelihood %s" %inv.get_GPR_model(inv.get_MOI(jitter), inv.zero_mean(2), data, test_grids)[0].log_marginal_likelihood().numpy())
            for invar_density in [20]:
                    try:
                        kernel = inv.get_SHM_Invariance(3, invar_density, jitter)
                        mean_function = inv.damping_SHM_mean(kernel, fixed, gamma, mass=1)
                        m, pred, var = inv.get_GPR_model(kernel, mean_function, data, test_grids)
                        print("density %s has %s log marginal likelihood"%(invar_density, m.log_marginal_likelihood().numpy()))

                    except tf.errors.InvalidArgumentError:
                        print("jitter too small")
                        continue
    print("\n")
