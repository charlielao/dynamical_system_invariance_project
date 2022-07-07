#%%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
plt.rcParams['text.usetex'] = True
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

