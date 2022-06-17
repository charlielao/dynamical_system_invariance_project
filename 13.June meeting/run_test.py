
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from termcolor import colored
import invariance_functions

test_grids = invariance_functions.get_test_points()
for jitter in [1e-8, 1e-7, 1e-6, 1e-5]:
    try:
        mean_function = invariance_functions.zero_mean(2)
        kernel = invariance_functions.MOI()
        print(invariance_functions.degree_of_freedom(kernel, test_grids))
#        get_GPR_model

    except tf.errors.InvalidArgumentError:
        print("%s jitter too small" %jitter)
        continue