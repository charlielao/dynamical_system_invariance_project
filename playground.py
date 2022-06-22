# %%
# %%
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, odeint
from scipy.misc import derivative
from scipy import interpolate
import random
from gpflow.utilities import print_summary, positive, to_default_float, set_trainable
from termcolor import colored
# %%
a = tf.Variable([0.1, 0.2, 0.3])
s = interpolate.BSpline([1, 2, 3, 4, 5, 6],a , 2)
# %%
b = tf.Variable([[1.,2.],[3.,4.]], dtype=tf.float64)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(a)
    tape.watch(b)
    c = tf.Variable(s(b))
print(tape.gradient(c, a))

    

# %%
