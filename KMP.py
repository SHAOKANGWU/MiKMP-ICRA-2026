# fast_kmp.py
import numpy as np

class KMP:
    """
    Python port of the C++ fast_kmp class.

    Data layout per row (same as your C++):
      [  inDim  |   outDim   |   outDim*outDim   ]
       inputs     outputs          C (per-row)

    Args
    ----
    oriData : (len, inDim + outDim + outDim*outDim) array
    inDim   : int
    outDim  : int
    pvFlag  : int   (0: position only, 1: position+velocity kernel)
    lamda   : float (regularization coefficient)
    kh      : float (RBF kernel width coefficient)
    thre    : float (distance threshold for replacement in insertPoint)
    """

    def __init__(self, oriData: np.ndarray, inDim: int, outDim: int,
                 pvFlag: int, lamda: float, kh: float, thre: float):
        self.data   = np.array(oriData, dtype=float, copy=True)
        self.len    = self.data.shape[0]
        self.inDim  = int(inDim)
        self.outDim = int(outDim)
        self.pvFlag = int(pvFlag)
        self.lamda  = float(lamda)
        self.kh     = float(kh)
        self.thre   = float(thre)

        # Will be filled by estimateMatrix()
        self.invK = None    # (len*outDim, len*outDim)
        self.W    = None    # (len*outDim, 1)

        self.estimateMatrix()

    def kernel_extend(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        kext = np.zeros((self.outDim, self.outDim), dtype=float)
        
        if self.pvFlag == 0:
            # RBF kernel k(a,b) * I_out
            diff = (a - b).reshape(-1)
            temp = np.exp(-self.kh * diff.dot(diff))
            np.fill_diagonal(kext, temp)

        elif self.pvFlag == 1:
            ta, tb = float(a[0]), float(b[0])
            dt     = 1e-4
            tadt   = ta + dt
            tbdt   = tb + dt

            kt_t       = np.exp(-self.kh * (ta - tb)   ** 2)      # k(ta, tb)
            kt_dt_temp = np.exp(-self.kh * (ta - tbdt) ** 2)      # k(ta, tb+dt)
            kt_dt      = (kt_dt_temp - kt_t) / dt

            kdt_t_temp = np.exp(-self.kh * (tadt - tb) ** 2)      # k(ta+dt, tb)
            kdt_t      = (kdt_t_temp - kt_t) / dt

            kdt_dt_temp = np.exp(-self.kh * (tadt - tbdt) ** 2)   # k(ta+dt, tb+dt)
            kdt_dt      = (kdt_dt_temp - kt_dt_temp - kdt_t_temp + kt_t) / (dt * dt)

            half = self.outDim // 2
            for i in range(half):
                kext[i, i]               = kt_t
                kext[i, i + half]        = kt_dt
                kext[half + i, i]        = kdt_t
                kext[half + i, half + i] = kdt_dt
        else:
            raise ValueError("pvFlag must be 0 or 1")

        return kext
    

    def estimateMatrix(self) -> int:
        """
        Build K (len*outDim x len*outDim) and Y (len*outDim x 1),
        then compute invK and W = invK @ Y.
        Mirrors C++ estimateMatrix().
        """
        if self.len <= 0:
            print("error1: initial kmp")
            return 0

        Y = np.zeros((self.len * self.outDim, 1), dtype=float)
        K = np.zeros((self.len * self.outDim, self.len * self.outDim), dtype=float)

        # Fill K and Y
        for i in range(self.len):
            ai = self.data[i, :self.inDim]
            # build C (outDim x outDim) from row i
            C = np.zeros((self.outDim, self.outDim), dtype=float)

            for j in range(self.len):
                bj = self.data[j, :self.inDim]
                temp = self.kernel_extend(ai, bj)  # outDim x outDim

                if i == j:
                    for k in range(self.outDim):
                        index = self.inDim + self.outDim + k * self.outDim
                        # row i, columns [index : index + outDim)
                        C[k, :] = self.data[i, index:index + self.outDim]
                    temp = temp + self.lamda * C

                K[i*self.outDim:(i+1)*self.outDim,
                  j*self.outDim:(j+1)*self.outDim] = temp

            # Y block from outputs at row i: columns [inDim : inDim+outDim)
            Y[i*self.outDim:(i+1)*self.outDim, 0] = self.data[i, self.inDim:self.inDim + self.outDim]

        # Invert and compute W
        self.invK = np.linalg.inv(K)
        self.W    = self.invK @ Y
        return 0
    

    def prediction(self, query: np.ndarray):
        """
        Predict mean and variance at 'query' (shape: (inDim,) or (inDim,1))
        Returns (mean: (outDim,), var: (outDim,outDim))
        Mirrors C++ prediction(); fixes a small bug by explicitly computing k(query,query).
        """
        q = query.reshape(-1)[:self.inDim]
        Ks = np.zeros((self.outDim, self.outDim * self.len), dtype=float)

        for i in range(self.len):
            si = self.data[i, :self.inDim]
            temp = self.kernel_extend(q, si)  # outDim x outDim
            Ks[:, i*self.outDim:(i+1)*self.outDim] = temp

        mean = (Ks @ self.W).reshape(-1)  # (outDim,)

        # k(query, query)
        kqq = self.kernel_extend(q, q)
        var = (self.len / self.lamda) * (kqq - Ks @ self.invK @ Ks.T)

        return mean, var
    

    # ---------------------------- updating the model ----------------------------
    def insertPoint_ori(self, newPoint: np.ndarray, updateFlag: bool) -> int:
        """
        Insert or replace nearest point (within 'thre'), then optionally recompute full model.
        Mirrors C++ insertPoint_ori().
        """
        newPoint = newPoint.reshape(-1)
        size = self.inDim + self.outDim * (self.outDim + 1)

        # Nearest in input space
        target = newPoint[:self.inDim]
        diffs  = self.data[:, :self.inDim] - target
        dists  = np.linalg.norm(diffs, axis=1)
        replaceIndex = int(np.argmin(dists))
        minDis       = float(dists[replaceIndex])
        replaceFlag  = 1 if minDis < self.thre else 0
        # print("replaceFlag:", replaceFlag)

        if replaceFlag:
            # Replace row
            self.data[replaceIndex, :size] = newPoint[:size]
        else:
            # Append row
            self.data = np.vstack([self.data, newPoint[:size]])
            self.len += 1

        # print("the new len is:", self.len)

        if updateFlag:
            self.estimateMatrix()

        return 0
    

    def insertPoint_fast(self, newPoint: np.ndarray) -> int:
        newPoint = newPoint.reshape(-1)
        size = self.inDim + self.outDim * (self.outDim + 1)

        via_input = newPoint[:self.inDim]
        via_var   = np.zeros((self.outDim, self.outDim), dtype=float)
        for i in range(self.outDim):
            index = self.inDim + self.outDim + i * self.outDim
            via_var[i, :] = newPoint[index:index + self.outDim]

        # Find nearest
        diffs  = self.data[:, :self.inDim] - via_input
        dists  = np.linalg.norm(diffs, axis=1)
        replaceIndex = int(np.argmin(dists))
        minDis       = float(dists[replaceIndex])
        replaceFlag  = 1 if minDis < self.thre else 0

        # If replacing: delete that point (update invK for deletion)
        if replaceFlag:
            # Remove row from data
            keep_mask = np.ones(self.len, dtype=bool)
            keep_mask[replaceIndex] = False
            self.data = self.data[keep_mask, :]
            self.len -= 1

            # indices to delete in the big K-inverse (outDim consecutive rows/cols)
            indices = np.arange(replaceIndex * self.outDim, (replaceIndex + 1) * self.outDim, dtype=int)
            self.inverse_after_delete(indices)
            # print("remove a datapoint, the new len is", self.len)

        # Update invK for adding the new point (block inverse augmentation)
        # Build B, C, D
        C = np.zeros((self.outDim, self.outDim * self.len), dtype=float)
        for i in range(self.len):
            temp = self.kernel_extend(via_input, self.data[i, :self.inDim])
            C[:, i*self.outDim:(i+1)*self.outDim] = temp
        B = C.T
        D = self.kernel_extend(via_input, via_input) + via_var

        self.augmented_matrix_inverse(B, C, D)

        # Append new point to data
        self.data = np.vstack([self.data, newPoint[:size]])
        self.len += 1
        # print("add a datapoint, the new len is", self.len)

        # Update W = invK * Y
        Y = np.zeros((self.len * self.outDim, 1), dtype=float)
        for i in range(self.len):
            Y[i*self.outDim:(i+1)*self.outDim, 0] = self.data[i, self.inDim:self.inDim + self.outDim]
        self.W = self.invK @ Y

        return 0


        # ---------------------------- inverse maintenance helpers ----------------------------
    def inverse_after_delete(self, indices: np.ndarray) -> int:
        """
        Update self.invK after deleting a block corresponding to row/col indices.
        Mirrors C++ inverse_after_delete(mat_inverse, indices).
        """
        ini_mat_inverse = self.invK
        a = np.delete(np.delete(ini_mat_inverse, indices, axis=0), indices, axis=1)
        b = np.delete(ini_mat_inverse[:, indices], indices, axis=0)  # 第i列
        c = np.delete(ini_mat_inverse[indices, :], indices, axis=1)  # 第i行
        d = ini_mat_inverse[np.ix_(indices, indices)]

        if np.linalg.det(d) == 0:
            raise ValueError('d is irreversible')

        d_inverse = np.linalg.inv(d)

        # Schur complement update
        self.invK = a - b @ d_inverse @ c
        return 0


    def augmented_matrix_inverse(self, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> int:
        """
        Update self.invK to the inverse of the augmented block matrix:
           inv([A  B; C  D]) given inv(A) = self.invK
        Mirrors C++ augmented_matrix_inverse().
        """
        A_inv = self.invK
        A_inv_B = A_inv @ B
        C_A_inv = C @ A_inv

        S = D - C @ A_inv_B
        S_inv = np.linalg.inv(S)

        top_left = A_inv + A_inv_B @ S_inv @ C_A_inv
        top_right = -A_inv_B @ S_inv
        bottom_left = -S_inv @ C_A_inv
        bottom_right = S_inv

        augmented_inv = np.block([[top_left, top_right],
                                [bottom_left, bottom_right]])
        self.invK = augmented_inv
        return 0
    
