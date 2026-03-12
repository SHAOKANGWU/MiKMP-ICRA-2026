import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import timeit
from functools import partial
import copy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

mat = loadmat('../../2Dletters/G.mat')
demosNum = 5 # number of demonstrations
dt = 0.005 # time interval
dataset = []
for i in range(demosNum):
    pos = mat['demos'][0, i]['pos'][0, 0]
    vel = np.gradient(pos, axis=-1) / dt
    pv = np.block([[pos], [vel]])
    dataset.append(pv)

dim = 2  # 2D trajectory
Dim = 2 * dim # both position and velocity
len = dataset[0].shape[1] # length of demonstrations
tau = dt * len # motion duration
t0 = np.linspace(dt, tau, num=len)
center = np.linspace(0, 1, 20) # centers of basis functions
basisNum = center.shape[0]  # number of basis functions
width = 50 # width of basis functions
lb1 = 1
lb2 = 10


def BasisFunction(t):
    Phi_1 = np.zeros(basisNum)
    addsum = 0
    for b in range(basisNum):
        Phi_1[b] = np.exp(-width * (t / tau - center[b]) ** 2)
        addsum = addsum + Phi_1[b]
    Phi_1 = Phi_1 / addsum
    Phi_2 = np.zeros(basisNum)
    addsum = 0
    for b in range(basisNum):
        Phi_2[b] = np.exp(-width * ((t + dt) / tau - center[b]) ** 2)
        addsum = addsum + Phi_2[b]
    Phi_2 = Phi_2 / addsum
    Phi_2 = (Phi_2 - Phi_1)/dt
    Phi_extend = np.zeros((Dim, basisNum*dim))
    for i in range(dim):
        Phi_extend[i, i*basisNum:(i+1)*basisNum] = Phi_1
    for i in range(dim):
        Phi_extend[dim+i, i * basisNum:(i+1)*basisNum] = Phi_2
    return Phi_extend


def update_w(t, desPoint, noisevar, mu_w, cov_w):
    Phi=BasisFunction(t)
    pred_pv = np.matmul(Phi,mu_w)
    sigma_Phi=np.matmul(cov_w, np.transpose(Phi))
    invL = np.linalg.pinv(np.matmul(Phi, sigma_Phi) + noisevar)
    sigma_Phi_invL = np.matmul(sigma_Phi, invL)
    mu_w_new = mu_w + np.matmul(sigma_Phi_invL, desPoint-pred_pv)
    cov_w_new = cov_w - np.matmul(sigma_Phi_invL, np.transpose(sigma_Phi))
    return mu_w_new, cov_w_new


Phi = np.zeros((Dim*len, basisNum*dim))
weights = np.zeros((demosNum, basisNum*dim))
mu_w = np.zeros(basisNum*dim)
cov_w = np.zeros((basisNum*dim, basisNum*dim))
via_var = 1E-6 * np.eye(Dim) # inversely proportional to the adaptation precision

for i in range(len):
    Phi[i*Dim:(i+1)*Dim, :] = BasisFunction(t0[i])
PhiT_Phi = np.matmul(np.transpose(Phi), Phi) + lb1 * np.eye(basisNum*dim)
cw = np.matmul(np.linalg.pinv(PhiT_Phi), np.transpose(Phi))

# Estimate the trajectory parameter vector w from each demonstration
for i in range(demosNum):
    expandedTraj = dataset[i].flatten(order='F') # concatenate all columns into a column vector
    weights[i, :] = np.matmul(cw, expandedTraj)

# Calculate the mean and covariance of w
mu_w = np.mean(weights, axis=0)
cov_w = np.matmul(np.transpose(weights - mu_w), weights - mu_w) / demosNum + lb2 * np.eye(basisNum*dim)

via_time1 = 0
via_point1 = np.array([9.77688474,  8.49074252, -6.32510836,  5.32676724]) # first desired point, format:[2D pos, 2D vel]
via_time2 = 0.245
via_point2 = np.array([-7.04454002,   4.74739098, -61.30230576, -72.87240131]) # second desired point, format:[2D pos, 2D vel]
via_time3 = 1.0
via_point3 = np.array([-1,  1.5, -1.46322122,  0]) # third desired point, format:[2D pos, 2D vel]
promp_constraints_list = [(via_time1, via_point1), (via_time2, via_point2), (via_time3, via_point3)]


#####################################################################################
#####################################################################################
################################# UPTIME TIME COST #################################
total_update_time = 0
for _ in range(1):
    
    new_mu_w = copy.deepcopy(mu_w)
    new_cov_w = copy.deepcopy(cov_w)

    ts = time.perf_counter()

    for constraint in promp_constraints_list:
        via_time, via_point = constraint
        # Update the distribution of w using each desired point
        [new_mu_w, new_cov_w] = update_w(via_time, via_point, via_var, new_mu_w, new_cov_w)
    
    te = time.perf_counter()
    total_update_time += (te - ts)
print("Average update time:", total_update_time / 1)
#####################################################################################
#####################################################################################
#####################################################################################



#####################################################################################
#####################################################################################
############################## INFERENCE TIME COST ##################################
pred_pv = np.zeros((Dim, len))
total_inference_time = 0
ts = time.perf_counter()
for i in range(len):
    pred_pv[:, i] = np.matmul(BasisFunction(t0[i]), new_mu_w)
te = time.perf_counter()
total_inference_time += (te - ts)
print("Average inference time:", total_inference_time / len)
print("length of trajectory:", len)
#####################################################################################
#####################################################################################
#####################################################################################



plt.plot(pred_pv[0], pred_pv[1], 'b', label='promp px-py', linewidth=2)
for i in range(demosNum):
   if i == 0:
       plt.plot(dataset[i][0], dataset[i][1], 'g--', linewidth=0.6, label='demos px-py')
   else:
       plt.plot(dataset[i][0], dataset[i][1], 'g--', linewidth=0.6)
plt.plot(via_point1[0], via_point1[1], 'go')
plt.plot(via_point2[0], via_point2[1], 'go')
plt.plot(via_point3[0], via_point3[1], 'go')
plt.xlabel("px")
plt.ylabel("py")
plt.legend()
plt.show()
ax1 = plt.subplot(2, 1, 1)
ax1.plot(t0, pred_pv[0], 'b', label='promp px', linewidth=2)
for i in range(demosNum):
    if i == 0:
        ax1.plot(t0, dataset[i][0], 'g--', linewidth=0.6, label='demos px')
    else:
        ax1.plot(t0, dataset[i][0], 'g--', linewidth=0.6)
plt.plot(via_time1, via_point1[0], 'go')
plt.plot(via_time2, via_point2[0], 'go')
plt.plot(via_time3, via_point3[0], 'go')
ax1.set_ylabel("px")
plt.legend()
ax1.grid()
ax2 = plt.subplot(2, 1, 2)
ax2.plot(t0, pred_pv[1], 'b', label='promp py', linewidth=2)
for i in range(demosNum):
    if i == 0:
        ax2.plot(t0, dataset[i][1], 'g--', linewidth=0.6, label='demos py')
    else:
        ax2.plot(t0, dataset[i][1], 'g--', linewidth=0.6)
plt.plot(via_time1, via_point1[1], 'go')
plt.plot(via_time2, via_point2[1], 'go')
plt.plot(via_time3, via_point3[1], 'go')
ax2.set_ylabel("py")
ax2.set_xlabel("t")
plt.legend()
ax2.grid()
plt.show()
ax1 = plt.subplot(2, 1, 1)
ax1.plot(t0, pred_pv[2], 'b', label='promp vx', linewidth=2)
for i in range(demosNum):
    if i == 0:
        ax1.plot(t0, dataset[i][2], 'g--', linewidth=0.6, label='demos vx')
    else:
        ax1.plot(t0, dataset[i][2], 'g--', linewidth=0.6)
plt.plot(via_time1, via_point1[2], 'go')
plt.plot(via_time2, via_point2[2], 'go')
plt.plot(via_time3, via_point3[2], 'go')
ax1.set_ylabel("vx")
plt.legend()
ax1.grid()
ax2 = plt.subplot(2, 1, 2)
ax2.plot(t0, pred_pv[3], 'b', label='promp vy', linewidth=2)
for i in range(demosNum):
    if i == 0:
        ax2.plot(t0, dataset[i][3], 'g--', linewidth=0.6, label='demos vy')
    else:
        ax2.plot(t0, dataset[i][3], 'g--', linewidth=0.6)
plt.plot(via_time1, via_point1[3], 'go')
plt.plot(via_time2, via_point2[3], 'go')
plt.plot(via_time3, via_point3[3], 'go')
ax2.set_ylabel("vy")
ax2.set_xlabel("t")
plt.legend()
ax2.grid()
plt.show()

np.save("saved_data/ProMP.npy", pred_pv.T)
