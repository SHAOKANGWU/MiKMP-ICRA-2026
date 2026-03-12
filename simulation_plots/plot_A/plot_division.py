import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from simulation_plots.plot_library import plot_gmm
from matplotlib.pyplot import MultipleLocator
import matplotlib.patheffects as pe

def plot_figure_kmp(ax, gmm_mu, gmm_cov, predict, via_points,
                    xlim_left, xlim_right, ylim_high, ylim_low, x_scale, y_scale, color):
    plot_gmm(gmm_mu, gmm_cov, swap=False, color=[255 / 255, 177 / 255, 50 / 255], alpha=0.03, ax=ax)
    ax.plot(predict[:, 0], predict[:, 1], label='predict', linewidth=6, color=color)
    ax.plot(via_points[0][0], via_points[0][1], ms=20, marker="o", color=[126 / 255, 47 / 255, 142 / 255],
            label="start-point")
    ax.plot(via_points[1][0], via_points[1][1], ms=20, marker="o", color=[0 / 255, 100 / 255, 200 / 255],
            label="via-point")
    ax.plot(via_points[2][0], via_points[2][1], ms=20, marker="o", color="g", label="end-point")
    ax.grid()
    ax.tick_params(width=1, labelsize=32)
    x_scale = MultipleLocator(x_scale)
    y_scale = MultipleLocator(y_scale)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_scale)
    ax.yaxis.set_major_locator(y_scale)
    plt.xlim(xlim_left, xlim_right)
    plt.ylim(ylim_low, ylim_high)


# mat = loadmat('../../2Dletters/refA.mat') # The reference trajectory is extracted from five demonstrations uisng GMM/GMR
mat = loadmat('../../2Dletters/refA.mat') # The reference trajectory is extracted from five demonstrations uisng GMM/GMR
dim = 2  # 2D trajectory
Dim = 2 * dim # both position and velocity
len = mat['refTraj'].shape[1] # length of the reference trajectory
seq_order = list(np.arange(len))

gmm_mu = np.zeros([dim, len])
gmm_cov = np.zeros([dim, dim, len])

for idx, order in enumerate(seq_order):
    gmm_mu[:, idx] = mat['refTraj'][0, order]['mu'][:dim, 0]
    gmm_cov[:, :, idx] = mat['refTraj'][0, order]['sigma'][:dim, :dim]


via_point1 = np.array([-9.67309744, -7.60580044, 13.24461949, 18.57873938])
via_point2 = np.array([1.93715746,  -3.47546397,  24.82473564, -58.39198364]) 
via_point3 = np.array([5.97952464, -1.44903256,  1.75837652, -1.36779445] )
via_point_list = [via_point1, via_point2, via_point3]

kmp_data = np.load("saved_data/KMP.npy")
ikmp_data = np.load("saved_data/iKMP.npy")
mikmp_1_data = np.load("saved_data/MiKMP_1.npy")
mikmp_2_data = np.load("saved_data/MiKMP_2.npy")

plt.figure(figsize=(12, 5))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

ax = plt.subplot(1, 2, 1)
plot_gmm(gmm_mu, gmm_cov, swap=False, color=[255 / 255, 177 / 255, 50 / 255], alpha=0.03, ax=ax)
ax.plot(gmm_mu[0], gmm_mu[1], linewidth=2, color=[255 / 255, 177 / 255, 50 / 255], linestyle='--')


# ax = plt.subplot(1, 2, 2)
ax.plot(kmp_data[:, 0], kmp_data[:, 1], label='KMP', linewidth=4, color=[0.6, 0.6, 0.6])
# ax.plot(ikmp_data[:, 0], ikmp_data[:, 1], label='iKMP', linewidth=2, color="#89ABE3", linestyle='--')
ax.plot(mikmp_1_data[:, 0], mikmp_1_data[:, 1], label='MiKMP-1', linewidth=5, color="#89ABE3")
ax.plot(mikmp_2_data[:, 0], mikmp_2_data[:, 1], label='MiKMP-2', linewidth=5, color=[252 / 255, 85 / 255, 84 / 255])

ax.plot(via_point_list[0][0], via_point_list[0][1], ms=10, marker="o", color=[126 / 255, 47 / 255, 142 / 255])
ax.plot(via_point_list[1][0], via_point_list[1][1], ms=10, marker="o", color=[0 / 255, 100 / 255, 200 / 255])
ax.plot(via_point_list[2][0], via_point_list[2][1], ms=10, marker="o", color="g")

ax.grid()
ax.tick_params(width=1, labelsize=20)
x_scale = MultipleLocator(10)
y_scale = MultipleLocator(10)
ax = plt.gca()
ax.xaxis.set_major_locator(x_scale)
ax.yaxis.set_major_locator(y_scale)
plt.xlim(-12, 12)
plt.ylim(-10, 10)
plt.legend()
plt.show()
