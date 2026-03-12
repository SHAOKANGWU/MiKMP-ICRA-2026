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



left_adap_real_traj = np.loadtxt("real_world_plots/real_traj/DualArm_Transportation_Adaptation/left_real_time.txt")
right_adap_real_traj = np.loadtxt("real_world_plots/real_traj/DualArm_Transportation_Adaptation/right_real_time.txt")
length = min(len(left_adap_real_traj), len(right_adap_real_traj))
half_length = length // 2 + 20

left_adap_real_traj = left_adap_real_traj[:length]
right_adap_real_traj = right_adap_real_traj[:length]


dt = 0.02
duration = len(left_adap_real_traj) * dt
tt = np.linspace(0, duration, len(left_adap_real_traj))
plt.plot(tt, left_adap_real_traj[:, 0], label='Left - x', color='b', linestyle='--')
plt.plot(tt, left_adap_real_traj[:, 1], label='Left - y', color='g', linestyle='--')
plt.plot(tt, left_adap_real_traj[:, 2], label='Left - z', color='r', linestyle='--')


duration = len(right_adap_real_traj) * dt
tt = np.linspace(0, duration, len(right_adap_real_traj))
plt.plot(tt, right_adap_real_traj[:, 0], label='Right - x', color='b')
plt.plot(tt, right_adap_real_traj[:, 1], label='Right - y', color='g')
plt.plot(tt, right_adap_real_traj[:, 2], label='Right - z', color='r')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title('Dual Arm Transportation - Adaptation')
plt.legend()
plt.grid()
plt.show()



# split adap to two parts. The first part is the reproduction, the second part is the adaptation
left_repro_real_traj = left_adap_real_traj[:half_length]
left_adap_real_traj = left_adap_real_traj[half_length:]
right_repro_real_traj = right_adap_real_traj[:half_length]
right_adap_real_traj = right_adap_real_traj[half_length:]

plt.plot(tt[:half_length], left_repro_real_traj[:, 0], label='Left Reproduction - x', color='b', linestyle='--')
plt.plot(tt[:half_length], left_repro_real_traj[:, 1], label='Left Reproduction - y', color='g', linestyle='--')
plt.plot(tt[:half_length], left_repro_real_traj[:, 2], label='Left Reproduction - z', color='r', linestyle='--')
plt.plot(tt[half_length:], left_adap_real_traj[:, 0], label='Left Adaptation - x', color='b')
plt.plot(tt[half_length:], left_adap_real_traj[:, 1], label='Left Adaptation - y', color='g')
plt.plot(tt[half_length:], left_adap_real_traj[:, 2], label='Left Adaptation - z', color='r')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title('Left Arm - Dual Arm Transportation')
plt.legend()
plt.grid()
plt.show()


# plot 3D trajectories
fig = plt.figure()
ax = plt.axes(projection="3d")
x_scale = 0.3
y_scale = 0.3
z_scale = 0.3
x_scale = MultipleLocator(x_scale)
y_scale = MultipleLocator(y_scale)
z_scale = MultipleLocator(z_scale)
ax.xaxis.set_major_locator(x_scale)
ax.yaxis.set_major_locator(y_scale)
ax.zaxis.set_major_locator(z_scale)
ax.set_xlim(-0.1, 0.8)
ax.set_ylim(-0.6, 0.6)
ax.set_zlim(-0.2, 0.6)
ax.tick_params(labelsize=18, width=11)
linewidth = 5
ax.view_init(azim=-115, elev=28)


# add start-point, via-point and end-point
left_repro_start = left_repro_real_traj[0]
left_repro_via = left_repro_real_traj[1150]
left_repro_end = left_repro_real_traj[-1]
left_adap_start = left_adap_real_traj[0]
left_adap_via = left_adap_real_traj[1150]
left_adap_end = left_adap_real_traj[-1]

right_repro_start = right_repro_real_traj[0]
right_repro_via = right_repro_real_traj[1150]
right_repro_end = right_repro_real_traj[-1]
right_adap_start = right_adap_real_traj[0]
right_adap_via = right_adap_real_traj[1150]
right_adap_end = right_adap_real_traj[-1]


ax.scatter(right_repro_via[0], right_repro_via[1], right_repro_via[2], s=80, marker="s", color=[0 / 255, 100 / 255, 200 / 255])
ax.scatter(right_repro_end[0], right_repro_end[1], right_repro_end[2], s=80, marker="s", color=[255 / 255, 177 / 255, 50 / 255])
ax.scatter(right_adap_via[0], right_adap_via[1], right_adap_via[2], s=80, marker="s", color=[126 / 255, 47 / 255, 142 / 255])
ax.scatter(right_adap_end[0], right_adap_end[1], right_adap_end[2], s=80, marker="s", color="g")

ax.scatter(left_repro_via[0], left_repro_via[1], left_repro_via[2], s=80, marker="s", color=[0 / 255, 100 / 255, 200 / 255])
ax.scatter(left_repro_end[0], left_repro_end[1], left_repro_end[2], s=80, marker="s", color=[255 / 255, 177 / 255, 50 / 255])
ax.scatter(left_adap_via[0], left_adap_via[1], left_adap_via[2], s=80, marker="s", color=[126 / 255, 47 / 255, 142 / 255])
ax.scatter(left_adap_end[0], left_adap_end[1], left_adap_end[2], s=80, marker="s", color="g")


ax.plot(left_repro_real_traj[:, 0], left_repro_real_traj[:, 1], left_repro_real_traj[:, 2], label='Left Reproduction',
        linewidth=4,  color=[252 / 255, 85 / 255, 84 / 255], linestyle="--")
ax.plot(left_adap_real_traj[:, 0], left_adap_real_traj[:, 1], left_adap_real_traj[:, 2], label='Left Adaptation',
        linewidth=4, color="#89ABE3",  #linestyle="--"
        )
   
ax.plot(right_repro_real_traj[:, 0], right_repro_real_traj[:, 1], right_repro_real_traj[:, 2], label='Right Reproduction',
        linewidth=4,  color=[252 / 255, 85 / 255, 84 / 255], linestyle="--")
ax.plot(right_adap_real_traj[:, 0], right_adap_real_traj[:, 1], right_adap_real_traj[:, 2], label='Right Adaptation',
        linewidth=4, color="#89ABE3",  #linestyle="--"
        )

plot_arrow(ax, 1900, left_repro_real_traj, color=[252 / 255, 85 / 255, 84 / 255], linestyle="solid")
# plot_arrow(ax, 3900, left_repro_real_traj, color=[252 / 255, 85 / 255, 84 / 255], linestyle="solid")

plot_arrow(ax, 700, left_adap_real_traj, color="#89ABE3", linestyle="solid")
plot_arrow(ax, 2200, left_adap_real_traj, color="#89ABE3", linestyle="solid")


plot_arrow(ax, 1900, right_repro_real_traj, color=[252 / 255, 85 / 255, 84 / 255], linestyle="solid")
# plot_arrow(ax, 3950, right_repro_real_traj, color=[252 / 255, 85 / 255, 84 / 255], linestyle="solid")

plot_arrow(ax, 500, right_adap_real_traj, color="#89ABE3", linestyle="solid")
plot_arrow(ax, 2200, right_adap_real_traj, color="#89ABE3", linestyle="solid")


plt.show()