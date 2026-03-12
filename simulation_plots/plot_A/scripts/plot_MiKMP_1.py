import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# set these before importing numpy/scipy
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from KMP import KMP
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
from itertools import repeat
import random
random.seed(0)
np.random.seed(0)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import copy

mat = loadmat('../../2Dletters/refA.mat') # The reference trajectory is extracted from five demonstrations uisng GMM/GMR
dim = 2  # 2D trajectory
Dim = 2 * dim # both position and velocity
len = mat['refTraj'].shape[1] # length of the reference trajectory
dt = 0.005 # time interval

tau = dt * len # motion duration
lb = 0.1  # regularization parameter
kh = 25  # kernel parameter
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

    gmm_mu[:, idx] = mat['refTraj'][0, order]['mu'][:dim, 0]
    gmm_cov[:, :, idx] = mat['refTraj'][0, order]['sigma'][:dim, :dim]

print(oriData.shape)

via_point1 = np.array([0.0, -9.67309744, -7.60580044, 13.24461949, 18.57873938] + via_var.flatten().tolist())
via_point2 = np.array([0.465, 1.93715746,  -3.47546397,  24.82473564, -58.39198364] + via_var.flatten().tolist())
via_point3 = np.array([1.0,  5.97952464, -1.44903256,  1.75837652, -1.36779445] + via_var.flatten().tolist())
kmp_constraints_list = [via_point1, via_point2, via_point3]

num_chunks = 4  # calculate number of chunks
data_chunks = np.array_split(oriData, num_chunks)
init_kmp_chunk = [KMP(oriData=chunk, inDim=1, outDim=Dim, pvFlag=1, lamda=lb, kh=kh, thre=0.001) for chunk in data_chunks]


#####################################################################################
#####################################################################################
################################# UPTIME TIME COST #################################
total_update_time = 0
for _ in range(1):
    kmp_chunk = copy.deepcopy(init_kmp_chunk)
    ts = time.perf_counter()

    for idx, kmp in enumerate(kmp_chunk):
        for via_point in kmp_constraints_list:
            kmp_chunk[idx].insertPoint_fast(via_point)
    
    te = time.perf_counter()
    total_update_time += te - ts
print("Average update time:", total_update_time / 1)
#####################################################################################
#####################################################################################
#####################################################################################




#####################################################################################
#####################################################################################
############################## INFERENCE TIME COST ##################################
start_time, end_time = 0, tau
pred_traj = []
total_inference_time, index = 0, 0

ts = time.perf_counter()
while start_time < end_time:
    query = np.array([start_time], dtype=float)
    weighted_mu_sum, precision_sum = np.zeros((Dim)), np.zeros((Dim, Dim))

    for j in range(num_chunks):
        mu, sigma = kmp_chunk[j].prediction(query)
        precision = np.linalg.pinv(sigma)  # or np.linalg.inv if well-conditioned

        weighted_mu_sum += precision @ mu
        precision_sum += precision

    fused_precision = np.linalg.pinv(precision_sum)
    fused_mean = fused_precision @ weighted_mu_sum
    pred_traj.append(fused_mean)

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
ax.grid()
plt.legend()
plt.show()

np.save("saved_data/MiKMP_1.npy", pred_traj)