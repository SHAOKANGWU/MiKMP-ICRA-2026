import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from KMP import KMP
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
import time
import copy

mat = loadmat('../../2Dletters/refA.mat') # The reference trajectory is extracted from five demonstrations uisng GMM/GMR
dim = 2  # 2D trajectory
Dim = 2 * dim # both position and velocity
len = mat['refTraj'].shape[1] # length of the reference trajectory
dt = 0.005 # time interval

tau = dt * len # motion duration
lb = 0.1  # regularization parameter
kh = 23  # kernel parameter
via_var = 1E-6 * np.eye(Dim) # inversely proportional to the adaptation precision
via_var[2, 2] = 1E4
via_var[3, 3] = 1E4

seq_order = list(np.arange(len))
# random.shuffle(seq_order)
oriData = np.zeros((len, 1 + Dim + Dim * Dim))
gmm_mu = np.zeros([dim, len])
gmm_cov = np.zeros([dim, dim, len])

for idx, order in enumerate(seq_order):
    oriData[idx, 0] = mat['refTraj'][0, order]['t']
    oriData[idx, 1:1+Dim] = mat['refTraj'][0, order]['mu'][:, 0]
    oriData[idx, 1+Dim:] = mat['refTraj'][0, order]['sigma'].flatten()

print(oriData.shape)

via_point1 = np.array([0.0, -9.67309744, -7.60580044, 13.24461949, 18.57873938] + via_var.flatten().tolist())
via_point2 = np.array([0.465, 1.93715746,  -3.47546397,  24.82473564, -58.39198364] + via_var.flatten().tolist())
via_point3 = np.array([1.0, 5.97952464, -1.44903256,  1.75837652, -1.36779445] + via_var.flatten().tolist())
kmp_constraints_list = [via_point1, via_point2, via_point3]
init_kmp = KMP(oriData=oriData, inDim=1, outDim=Dim, pvFlag=1, lamda=lb, kh=kh, thre=0.001)


#####################################################################################
#####################################################################################
################################# UPTIME TIME COST #################################
total_update_time = 0
for _ in range(1):
    kmp = copy.deepcopy(init_kmp)
    
    ts = time.perf_counter()
    for via_point in kmp_constraints_list:
        kmp.insertPoint_fast(via_point)
    te = time.perf_counter()
    total_update_time += te - ts
print("Average update time:", total_update_time / 1)
#####################################################################################
#####################################################################################
#####################################################################################



#####################################################################################
#####################################################################################
############################## INFERENCE TIME COST ##################################
start_time, end_time, pred_traj = 0, tau, []
total_inference_time, index = 0, 0

ts = time.perf_counter()
while start_time < end_time:
    query = np.array(start_time)
    mu, sigma = kmp.prediction(query)
    pred_traj.append(mu)
    start_time += dt
    index += 1
te = time.perf_counter()

total_inference_time += (te - ts)
print("Average inference time:", total_inference_time / index)
print("length of trajectory:", index)
pred_traj = np.array(pred_traj)
#####################################################################################
#####################################################################################
#####################################################################################


# Visualization
ax = plt.subplot(1, 1, 1)
ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', label='KMP Prediction')
for via_point in kmp_constraints_list:
    ax.plot(via_point[1], via_point[2], 'go', markersize=8, label='Via Point')

plt.show()

np.save("saved_data/iKMP.npy", pred_traj)