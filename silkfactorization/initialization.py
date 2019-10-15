import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import logging
import numpy.random as npr

from . import kalman_parameters

ASL=sp.sparse.linalg.aslinearoperator
logger = logging.getLogger(__name__)

def initialize_from_scipy_sparse(data,Nk,ds_U,ds_alpha,MU,Malpha):
    muhat_U,muhat_alpha = get_initial_muhat_guess_from_scipy_sparse(data,Nk,MU,Malpha)
    kp_U=initialize_kp(muhat_U,ds_U,MU)
    kp_alpha=initialize_kp(muhat_alpha,ds_alpha,Malpha)
    return dict(kp_U=kp_U,kp_alpha=kp_alpha)

def initialize_kp(muhat,ds,M):
    sighat = np.abs(muhat)/3 + .1

    mu=np.mean(muhat,axis=0)
    sigma=np.std(muhat,axis=0)

    assert (ds>0).all()
    rho=np.ones(muhat.shape[1])*(1.0/np.min(ds))*(1+npr.rand(muhat.shape[1])*.1)

    return dict(
        muhat=muhat,
        rho=rho,
        mu=mu,
        sigma=sigma,
        sighat=sighat,
        ds=np.r_[np.inf,ds,np.inf],
        M=M
    )

def pge_safe_justone(x):
    x=np.abs(x)
    if x>.000001:
        return np.tanh(x/2)/(2*x)
    else:
        return .25-0.020833333333333332*(x**2)

def make_dmhalf(data):
    assert sp.sparse.issparse(data)
    Nr,Nc = data.shape
    one_Nr=ASL(np.ones((Nr,1)))
    one_Nc=ASL(np.ones((Nc,1)))
    return ASL(data) - .5*(one_Nr@one_Nc.H)

def get_initial_muhat_guess_from_scipy_sparse(data,Nk,MU,Malpha):
    '''
    mtx is scipy sparse matrix
    '''

    assert sp.sparse.issparse(data)
    data=data.tocsr()
    Nr,Nc=data.shape

    # get the fully observed portion
    fully_observed = data[~MU][:,~Malpha]
    partial_columns = data[~MU][:,Malpha]
    partial_rows = data[MU][:,~Malpha]

    # do SVD on it
    logger.info("svd for initialization")
    U,e,alpha = sp.sparse.linalg.svds(make_dmhalf(fully_observed),Nk)
    U = U @ np.diag(np.sqrt(e))
    alpha = (alpha.T) @ np.diag(np.sqrt(e))

    # get unobserved alphas 
    mtx = U.T @ U  # Nk x Nk
    vec = (make_dmhalf(partial_columns).H @ U).T # Nk x partial columns
    unobserved_alpha = np.linalg.solve(mtx,vec).T

    # get unobserved Us
    mtx = alpha.T @ alpha  # Nk x Nk
    vec = make_dmhalf(partial_rows) @ alpha # partial rows x Nk
    unobserved_U = np.linalg.solve(mtx,vec.T).T

    # put it together
    final_U=np.zeros((Nr,Nk))
    final_alpha=np.zeros((Nc,Nk))
    final_U[~MU]=U
    final_U[MU]=unobserved_U
    final_alpha[~Malpha]=alpha
    final_alpha[Malpha]=unobserved_alpha

    return 2*final_U,2*final_alpha