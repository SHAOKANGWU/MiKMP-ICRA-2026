import numpy as np

class NS_KMP:
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

        # For null space adaptation
        self.K_hat = None
        self.K_underline_inv = None
        self.xi_hat = None
        self.nb_newPoints = None
        self.newPoints_list = None

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
    

    def prediction_nullspace(self, query: np.ndarray):
        q = query.reshape(-1)[:self.inDim]
        Ks = np.zeros((self.outDim, self.outDim * self.len), dtype=float)

        for i in range(self.len):
            si = self.data[i, :self.inDim]
            temp = self.kernel_extend(q, si)  # outDim x outDim
            Ks[:, i*self.outDim:(i+1)*self.outDim] = temp

        kmp_prediction = (Ks @ self.W).reshape(-1)  # (outDim,)

        # compute k_star_hat
        k_star_hat = np.zeros((self.outDim, self.outDim * self.nb_newPoints), dtype=float)
        for i in range(self.nb_newPoints):
            s_via = self.newPoints_list[i][:self.inDim]
            temp = self.kernel_extend(q, s_via)  # outDim x outDim
            k_star_hat[:, i*self.outDim:(i+1)*self.outDim] = temp

        # null space modulation
        null_space_modulation = ( k_star_hat - Ks @ self.invK @ self.K_hat ) @ self.K_underline_inv @ self.xi_hat 
        null_space_modulation = null_space_modulation.reshape(-1)
        mean = kmp_prediction + null_space_modulation

        return mean, None  # var is not computed in null space prediction


    def insertPoint_nullspace(self, newPoints_list: list) -> int:
        """newPoints_list: list of array, each of shape (inDim + outDim + outDim*outDim,)"""
        
        nb_newPoints = len(newPoints_list)
        K_hat = np.zeros((self.len * self.outDim, nb_newPoints * self.outDim), dtype=float)
        K_underline = np.zeros((self.outDim * nb_newPoints, self.outDim * nb_newPoints), dtype=float)
        xi_hat = np.vstack([newPoints_list[i][self.inDim:self.inDim + self.outDim].reshape(1, -1) for i in range(nb_newPoints)])
 
        # compute K_hat
        for i in range(self.len):
            s_in = self.data[i, :self.inDim]

            for j in range(nb_newPoints):
                s_via = newPoints_list[j][:self.inDim]

                temp = self.kernel_extend(s_in, s_via)  # outDim x outDim

                K_hat[i*self.outDim:(i+1)*self.outDim,
                  j*self.outDim:(j+1)*self.outDim] = temp
        

        # compute K_underline
        for i in range(nb_newPoints):
            s_via_i = newPoints_list[i][:self.inDim]

            for j in range(nb_newPoints):
                s_via_j = newPoints_list[j][:self.inDim]
                
                temp = self.kernel_extend(s_via_i, s_via_j)  # outDim x outDim
                K_underline[i*self.outDim:(i+1)*self.outDim,
                            j*self.outDim:(j+1)*self.outDim] = temp
                
                # if i == j:
                #     cov = np.zeros((self.outDim, self.outDim), dtype=float)
                #     for k in range(self.outDim):
                #         index = self.inDim + self.outDim + k * self.outDim
                #         # row i, columns [index : index + outDim)
                #         cov[k, :] = newPoints_list[i][index:index + self.outDim]
                #     temp = temp + self.lamda * cov


        K_ZX = K_hat.T
        K_underline = K_underline - K_ZX @ (self.invK @ K_hat)
    

        self.K_hat = K_hat
        self.K_underline_inv = np.linalg.inv(K_underline)
        self.xi_hat = xi_hat.reshape(-1, 1)
        self.nb_newPoints = nb_newPoints
        self.newPoints_list = newPoints_list

        return 0