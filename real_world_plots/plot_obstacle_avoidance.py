import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# 3D trajectory
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_arrow(ax, point_index, traj, color, linestyle):
    dx, dy, dz = np.gradient(traj[:, 0]), np.gradient(traj[:, 1]), np.gradient(traj[:, 2])
    ax.arrow3D(traj[point_index, 0], traj[point_index, 1], traj[point_index, 2],
               dx[point_index], dy[point_index], dz[point_index],
               mutation_scale=25,
               arrowstyle="->",
               linestyle=linestyle,
               color=color,
               linewidth=4
               )

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
setattr(Axes3D, 'arrow3D', _arrow3D)



repro_real_traj = np.loadtxt("real_world_plots/real_traj/Obstacle_Avoidance_reproduction/real_trajectory.txt")
adap_real_traj = np.loadtxt("real_world_plots/real_traj/Obstacle_Avoidance_adaptation/real_trajectory.txt")


# TODO: plot start point and end point, also plot the arrow to show the direction of the trajectory
dt = 0.02
duration = len(adap_real_traj) * dt

# plot t-x, t-y, t-z in one figure
tt = np.linspace(0, duration, len(adap_real_traj))
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
plt.plot(tt, adap_real_traj[:, 0], label='Adaptation x', linestyle="dashed")
plt.plot(tt, adap_real_traj[:, 1], label='Adaptation y', linestyle="dashed")
plt.plot(tt, adap_real_traj[:, 2], label='Adaptation z', linestyle="dashed")

duration = len(repro_real_traj) * dt
tt = np.linspace(0, duration, len(repro_real_traj))
plt.plot(tt, repro_real_traj[:, 0], label='Reproduction x')
plt.plot(tt, repro_real_traj[:, 1], label='Reproduction y')
plt.plot(tt, repro_real_traj[:, 2], label='Reproduction z')

ax.fill_between([6.25, 8.25], -0.6, 0.8, facecolor=[0.6, 0.6, 0.6], alpha=0.3)
ax.fill_between([20.6, 22.6], -0.6, 0.8, facecolor=[0.6, 0.6, 0.6], alpha=0.3)

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Adaptation Trajectory")
plt.grid()
plt.show()


via_idx1 = int(6.25 / dt)
via_idx2 = int(8.25 / dt)
via_idx3 = int(20.6 / dt)
via_idx4 = int(22.6 / dt)

first_guidance_idx_1 = via_idx1
first_guidance_idx_2 = via_idx2
second_guidance_idx_1 = via_idx3
second_guidance_idx_2 = via_idx4

# plot the 3D trajectories inside the trajectories list
fig = plt.figure()
ax = plt.axes(projection="3d")
x_scale = 0.3
y_scale = 0.4
z_scale = 0.3
x_scale = MultipleLocator(x_scale)
y_scale = MultipleLocator(y_scale)
z_scale = MultipleLocator(z_scale)
ax.xaxis.set_major_locator(x_scale)
ax.yaxis.set_major_locator(y_scale)
ax.zaxis.set_major_locator(z_scale)
ax.set_xlim(-0.2, 0.8)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.2, 0.6)
ax.tick_params(labelsize=18, width=11)
linewidth = 5
ax.view_init(azim=6, elev=33)

ax.plot(repro_real_traj[:, 0], repro_real_traj[:, 1], repro_real_traj[:, 2], label='Reproduction',
        linewidth=6,  color="#89ABE3", linestyle="--")  

ax.plot(adap_real_traj[:, 0], adap_real_traj[:, 1], adap_real_traj[:, 2], label='Adaptation', 
         linewidth=6, color=[252 / 255, 85 / 255, 84 / 255], 
         )


# plot the via-points
ax.scatter(adap_real_traj[first_guidance_idx_1, 0], adap_real_traj[first_guidance_idx_1, 1], adap_real_traj[first_guidance_idx_1, 2],
           color=[126 / 255, 47 / 255, 142 / 255], s=120, label='Guidance 1 - Via-point 1') 
ax.scatter(adap_real_traj[first_guidance_idx_2, 0], adap_real_traj[first_guidance_idx_2, 1], adap_real_traj[first_guidance_idx_2, 2],
              color=[126 / 255, 47 / 255, 142 / 255], s=120, label='Guidance 1 - Via-point 2')
ax.scatter(adap_real_traj[second_guidance_idx_1, 0], adap_real_traj[second_guidance_idx_1, 1], adap_real_traj[second_guidance_idx_1, 2],
                color="g", s=120, label='Guidance 2 - Via-point 1')
ax.scatter(adap_real_traj[second_guidance_idx_2, 0], adap_real_traj[second_guidance_idx_2, 1], adap_real_traj[second_guidance_idx_2, 2],
                color="g", s=120, label='Guidance 2 - Via-point 2')

# ax.legend(ncol=3, fontsize=13)

plot_arrow(ax, 380, adap_real_traj, color=[252 / 255, 85 / 255, 84 / 255], linestyle="solid")
plot_arrow(ax, 800, adap_real_traj, color=[252 / 255, 85 / 255, 84 / 255], linestyle="solid")
plot_arrow(ax, 1098, adap_real_traj, color=[252 / 255, 85 / 255, 84 / 255], linestyle="solid")
# plot_arrow(ax, 1900, adap_real_traj, color=[252 / 255, 85 / 255, 84 / 255], linestyle="solid")
plot_arrow(ax, 1300, adap_real_traj, color=[252 / 255, 85 / 255, 84 / 255], linestyle="solid")


# plt.savefig("tmp")
plt.show()
