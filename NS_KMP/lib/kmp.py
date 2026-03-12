import numpy as np
from sklearn import mixture
from gmr import *

class kmp:
    "A class with basic functionalities of kernelized movement primitives."

    def __init__(self, data=[], Kgmm=5, N=100, l=0.1, h=1.0, lambda1=0.1, lambda2=100, alpha=100):
        self.Kgmm = Kgmm
        self.N = N
        self.l = l
        self.h = h
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.data = data
        self.model_in = []
        self.model = []
        self.mu = []
        self.sigma = []
        self.mu_block = []
        self.sigma_block = []
        self.K = []
        self.K_s = []
        self.K_ss = []
        self.nvia = 0                       # start with zero via-points

        print("KMP with:",\
        "\n\tK = ", Kgmm,\
        "\n\tN = ", N,\
        "\n\tl = ", l,\
        "\n\tlambda1 = ", lambda1,\
        "\n\tlambda2 = ", lambda2,\
        "\n\talpha = ", alpha)

    def init_ref_traj(self, data, nb_dim_in, nb_dim_out, x_in=None):
                
        # Train GMM on demonstration data
        gmm = mixture.GaussianMixture(n_components=self.Kgmm,covariance_type='full',reg_covar=1E-5,init_params='kmeans')
        gmm.fit(data)
        
        # GMR
        d_in = range(0,nb_dim_in)
        d_out = range(nb_dim_in,nb_dim_in+nb_dim_out)
        x_in = np.linspace(0,1,self.N)
        mu, sigma = gmr(gmm, x_in, d_in, d_out, self.N)
        
        # Block-ize GMR output
        mu_block = mu.reshape(-1,1)                         # mean
        pntr_sigma = []                  
        for i in range(0,sigma.shape[0]):                   # covariance
            pntr_sigma.append(sigma[i,:,:])
        sigma_block = sp.block_diag(*pntr_sigma)
        
        self.nb_dim_in = nb_dim_in
        self.nb_dim_out = nb_dim_out
        self.x_in = x_in.reshape(-1,nb_dim_in) # need a 2D array because input might be multi-dim
        self.model = gmm                       # this is kept for analysis, could be discarded otherwise
        self.mu = mu
        self.sigma = sigma
        self.mu_block = mu_block
        self.sigma_block = sigma_block
        self.update_K()

    def rbf_kernel(self, x_i,x_j,l=0.1,h=1.0):
        dif = x_i-x_j
        sq_dist = np.dot(dif,dif)
    #     k = h*np.exp(-l*dist)
        k = h*np.exp(-sq_dist/(2*l**2))
        return k

    def matern_kernel_p0(self, x_i,x_j,l=0.1,h=1.0):
        dif = x_i-x_j
        dist = np.sqrt(np.dot(dif,dif))
        k = h*np.exp(-dist/l)
        return k

    def matern_kernel_p1(self, x_i,x_j,l=0.1,h=1.0):
        dif = x_i-x_j
        dist = np.sqrt(np.dot(dif,dif))
        k = h*(1+np.sqrt(3)*dist/l)*np.exp(-np.sqrt(3)*dist/l)
        return k

    def matern_kernel_p2(self, x_i,x_j,l=0.1,h=1.0):
        dif = x_i-x_j
        dist = np.sqrt(np.dot(dif,dif))
        k = h*(1 + np.sqrt(5)*dist/l + 5*dist**2/(3*l**2))*np.exp(-np.sqrt(5)*dist/l)
        return k

    def kernel_matrix(self, x1, x2, l, h):
        K = np.zeros((x1.shape[0],x2.shape[0]))
        for i in range(0,x1.shape[0]):
            for j in range(0,x2.shape[0]):
                K[i,j] = self.rbf_kernel(x1[i,:],x2[j,:],l,h)
    #             K[i,j] = matern_kernel_p0(x1[i,:],x2[j,:],l,h)
    #             K[i,j] = matern_kernel_p1(x1[i,:],x2[j,:],l,h)
    #             K[i,j] = matern_kernel_p2(x1[i,:],x2[j,:],l,h)
        return K            
    
    def update_K(self):
        self.K = np.kron(self.kernel_matrix(self.x_in, self.x_in, self.l, self.h), np.eye(self.nb_dim_out,self.nb_dim_out)) # K
       
    def update_inputs(self, x_test):
        """ Computes K_s and K_ss, the kernel matrices that depend on newly observed inputs"""
        self.x_test = x_test
        I_O = np.eye(self.nb_dim_out, self.nb_dim_out)
        self.K_s = np.kron(self.kernel_matrix(x_test.reshape(-1, self.nb_dim_in), self.x_in, self.l, self.h), I_O)
        self.K_ss = np.kron(self.kernel_matrix(x_test.reshape(-1, self.nb_dim_in), x_test.reshape(-1, self.nb_dim_in),self.l, self.h), I_O)
        #print self.K
        #print self.K_s
        #print self.K_ss

    def mean(self, x_test):
        #K_s = np.kron(self.kernel_matrix(x_test.reshape(-1, self.nb_dim_in), self.x_in),
         #             np.eye(self.nb_dim_out, self.nb_dim_out))
        #K_ss = np.kron(self.kernel_matrix(x_test.reshape(-1, self.nb_dim_in), x_test.reshape(-1, self.nb_dim_in)),
          #             np.eye(self.nb_dim_out, self.nb_dim_out))
        #print(self.K.shape)
        #print(self.K_s.shape)
        #print(self.sigma_block.shape)
        self.mu_out = self.K_s.dot(np.linalg.inv(self.K + self.lambda1 * self.sigma_block)).dot(self.mu_block)
        #print self.mu_out
        return self.mu_out

    def cov(self, x_test):
        #K_s = np.kron(self.kernel_matrix(x_test.reshape(-1, self.nb_dim_in), self.x_in),
           #           np.eye(self.nb_dim_out, self.nb_dim_out))
        #K_ss = np.kron(self.kernel_matrix(x_test.reshape(-1, self.nb_dim_in), x_test.reshape(-1, self.nb_dim_in)),
         #              np.eye(self.nb_dim_out, self.nb_dim_out))
        #self.Sigma_test = (self.N / self.lambda2) * (self.K_ss - self.K_s.dot(np.linalg.inv(self.K + self.lambda2 * self.sigma_block)).dot(self.K_s.T))
        self.Sigma_test = self.alpha * (self.K_ss - self.K_s.dot(np.linalg.inv(self.K + self.lambda2 * self.sigma_block)).dot(self.K_s.T))
        return self.Sigma_test

    def predict(self, x_test):
        self.mean(x_test)
        self.cov(x_test)
        return [self.mu_out, self.Sigma_test]

    def add_viapoints(self, input_via, output_via, gamma=1000):
        "'gamma' assigns the importance of the via-points (the higher, the more important and the trajectory will pass through)"

        I_O = np.eye(self.nb_dim_out, self.nb_dim_out)
        via_point = [input_via, output_via, gamma * I_O]     # list of np.arrays: [np.array, np.array, np.array]

        ### Recompute K, k*, k**
        self.x_in = np.append(self.x_in, via_point[0])  # crucial step 1)
        self.x_in = self.x_in.reshape(-1, self.nb_dim_in)
        self.mu_block = np.append(self.mu_block, via_point[1].T)        # crucial step 2)
        self.sigma = np.append(self.sigma, via_point[2].reshape(1, self.nb_dim_out, -1), axis=0)    # crucial step 3)
        self.nvia += 1

        # Blockize GMR output + the new covariance
        import scipy.linalg as sp
        pntr_sigma = []
        for i in range(0, self.sigma.shape[0]):
            pntr_sigma.append(self.sigma[i, :, :])
        self.sigma_block = sp.block_diag(*pntr_sigma)
        self.update_K()
        self.update_inputs(self.x_test)


    # def reinit_traj(self):

    # def remove_via_point(self):
