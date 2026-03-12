import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_2D_gaussian(mu, cov, ax, color='r', alpha=0.2):
    
    '''Plot a 2x2 Gaussian distribution'''

    # adapted from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mu[:2], v[0], v[1], 180 + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(alpha)
    ax.add_artist(ell)
#     ax.set_aspect('equal')

def plot_gmm_2d(mu, cov, ax, color='r', alpha=0.2):
    
    '''2D plot of all K components of a GMM with 2x2 Gaussian distributions
    cov: K x 2 x 2 array
    mu: K x 2 array
    ax: axes of 2d plot'''

    for cov_, mu_ in zip(cov,mu):
        plot_2D_gaussian(cov_, mu_, ax, color, alpha)

def plot_gmm_2d_sklearn(gmm, ax, color='r', alpha=0.2):
    
    '''2D plot of all components of a GMM using a GaussianMixture object from scikit-learn
    gmm: object of sklearn.mixture.GaussianMixture class
    ax: axes of 2d plot'''
    
    cov = np.zeros((gmm.n_components,2,2))
    if gmm.covariance_type == 'full':
        cov = gmm.covariances_
    elif gmm.covariance_type == 'tied':
        for i in range(0,gmm.n_components):
            cov[i,:,:]=gmm.covariances_
    elif gmm.covariance_type == 'diag':
        for i, cov_ in enumerate(gmm.covariances_):
            cov[i,:,:]=np.eye(2)*np.diag(cov_)
    elif gmm.covariance_type == 'spherical':
        for i, cov_ in enumerate(gmm.covariances_):
            cov[i,:,:]=np.eye(2)*cov_

    plot_gmm_2d(gmm.means_, cov, ax, color, alpha)

def plot_3D_gaussian(ell_matrix, center, ax, color='r', alpha=0.2):

    '''Plot a 3x3 Gaussian distribution'''
    
    # use eigenvalue decomposition to find the scaling and rotation of each axis
    s, rotation = np.linalg.eig(ell_matrix)
    
    # rearrange axes by their magnitude
    sort_ix = np.argsort(s)
    s = s[sort_ix]
    rot_tmp = np.copy(rotation.T)
    rotation = np.copy(rot_tmp[sort_ix].T) # maybe there's a more elegant way to sort matrix columns
    
    # adapted from https://stackoverflow.com/a/14958796
    radii= np.sqrt(s) # rescale by integer to plot N stdevs
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot(rotation,[x[i,j],y[i,j],z[i,j]]) + center

    # plot
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, alpha=alpha)

def plot_gmm_3d(cov,mu,ax, color='r', alpha=0.2):
    
    '''3D plot of all K components of a GMM with 3x3 Gaussian distributions
    cov: K x 3 x 3 array
    mu: K x 3 array
    ax: axes of 3d plot'''
    
    for cov_, mu_ in zip(cov,mu):
        plot_3D_gaussian(cov_, mu_, ax, color, alpha)

def plot_gmm_3d_sklearn(gmm, ax, color='r', alpha=0.2):
    
    '''3D plot of all components of a GMM using a GaussianMixture object from scikit-learn
    gmm: object of sklearn.mixture.GaussianMixture class
    ax: axes of 3d plot'''
    
    cov = np.zeros((gmm.n_components,3,3))
    if gmm.covariance_type == 'full':
        cov = gmm.covariances_
    elif gmm.covariance_type == 'tied':
        for i in range(0,gmm.n_components):
            cov[i,:,:]=gmm.covariances_
    elif gmm.covariance_type == 'diag':
        for i, cov_ in enumerate(gmm.covariances_):
            cov[i,:,:]=np.eye(3)*np.diag(cov_)
    elif gmm.covariance_type == 'spherical':
        for i, cov_ in enumerate(gmm.covariances_):
            cov[i,:,:]=np.eye(3)*cov_

    plot_gmm_3d(cov, gmm.means_, ax, color, alpha)