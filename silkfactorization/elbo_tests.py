import numpy as np
import numpy.linalg
import types
import logging
import numpy.random as npr
import numpy as np
import scipy as sp
import tensorflow as tf

from . import initialization
from . import big_linear_operators
from . import binary_sparse_matrix
from . import training
from . import kalman_parameters

logger = logging.getLogger(__name__)

def pge_safe_justone(x):
    x=np.abs(x)
    if x>.000001:
        return np.tanh(x/2)/(2*x)
    else:
        return .25-0.020833333333333332*(x**2)

def get_G(sp):
    ekp_U=ExplodedKP(sp.kp_U)
    ekp_alpha=ExplodedKP(sp.kp_alpha)
    Nr=ekp_U.muhat.shape[0]
    Nc=ekp_alpha.muhat.shape[0]
    G=np.zeros((Nr,Nc))
    for r in range(Nr):
        for c in range(Nc):
            G1 = np.sum(ekp_U.muhat[r]*ekp_alpha.muhat[c])**2
            G2 = np.sum((ekp_U.sighat[r]**2)*(ekp_alpha.sighat[c]**2))
            G3 = np.sum((ekp_U.sighat[r]**2)*(ekp_alpha.muhat[c]**2))
            G4 = np.sum((ekp_U.muhat[r]**2)*(ekp_alpha.sighat[c]**2))
            Gsq=G1+G2+G3+G4
            G[r,c]=np.sqrt(Gsq)
    return G

def data_loss(X,sp):
    X=np.array(X)
    ekp_U=ExplodedKP(sp.kp_U)
    ekp_alpha=ExplodedKP(sp.kp_alpha)
    Nr,Nc=X.shape

    logits = ekp_U.muhat @ ekp_alpha.muhat.T

    G=get_G(sp)

    mask = 1- ekp_U.M[:,None]*ekp_alpha.M[None,:]
    loss1 = -np.sum((X-.5) * logits * mask)
    loss2 = np.sum(np.log(2*np.cosh(.5*G))*mask)
    loss = loss1+loss2

    logger.debug("data loss = "+str(loss1))
    logger.debug("cosh_loss = "+str(loss2))

    return np.sum(loss)


class ExplodedKP:
    def __init__(self,kp):
        dct=kp.dump()
        for x in dct:
            setattr(self,x,dct[x])

        self.length = self.muhat.shape[0]
        self.Nk=int(self.mu.shape[0])

        self.sigsq=self.sigma**2
        self.sigsqi=1.0/self.sigsq


        self.big_Sigma=np.zeros((self.length,self.Nk,self.length,self.Nk))
        for k in range(self.Nk):
            for i in range(self.length):
                for j in range(i,self.length):
                    # what is distance from i to j?
                    dst = np.sum(self.ds[i+1:j+1])
                    self.big_Sigma[i,k,j,k]=(self.sigma[k]**2) * np.exp(-.5 * self.rho[k] * dst)
                    self.big_Sigma[j,k,i,k]=self.big_Sigma[i,k,j,k]

        self.big_Sigma_raveled = self.big_Sigma.reshape((self.length*self.Nk,self.length*self.Nk))
        self.big_Sigmai_raveled=np.linalg.inv(self.big_Sigma_raveled)
        self.big_Sigmai=self.big_Sigmai_raveled.reshape(self.big_Sigma.shape)

        self.big_Sigmahat = np.zeros((self.length,self.Nk,self.length,self.Nk))
        for k in range(self.Nk):
            for i in range(self.length):
                self.big_Sigmahat[i,k,i,k]= self.sighat[i,k]**2
        self.big_Sigmahat_raveled = self.big_Sigmahat.reshape(self.big_Sigma_raveled.shape)

    def apply_big_Sigmai(self,matrix):
        return (self.big_Sigmai_raveled @ matrix.ravel()).reshape(matrix.shape)

    def KL(self):
        trace_term= np.sum(self.big_Sigmai*self.big_Sigmahat)

        df = self.mu[None,:] - self.muhat
        mu_term = np.sum(df * self.apply_big_Sigmai(df))

        constant_term = -self.length*self.Nk

        det_Sig = np.linalg.slogdet(self.big_Sigma_raveled)[1]
        neg_det_Sighat = -np.linalg.slogdet(self.big_Sigmahat_raveled)[1]

        logger.debug('NP: trace_term = '+str(trace_term))
        logger.debug('NP: mu_term = '+str(mu_term))
        logger.debug('NP: constant_term = '+str(constant_term))
        logger.debug('NP: neg_det_Sighat = '+str(neg_det_Sighat))
        logger.debug('NP: det_Sig = '+str(det_Sig))

        return .5*(trace_term+mu_term+constant_term+neg_det_Sighat+det_Sig)

###################


def test():
    Nr=40
    Nc=38
    Nk=2
    U=npr.randn(Nr,Nk)
    alpha=npr.randn(Nc,Nk)
    logits = U@alpha.T 

    mtx=1.0*(npr.rand(Nr,Nc)<sp.special.expit(logits))

    ds_U=npr.rand(Nr-1)+.1
    ds_alpha=npr.rand(Nc-1)+.1

    heldout_rows=np.r_[0:Nr]>20
    heldout_columns=np.r_[0:Nc]>21


    mtx=sp.sparse.coo_matrix(mtx)

    ##################
    snap=training.make_initial_snapshot(mtx,2,ds_U,ds_alpha,heldout_rows,heldout_columns)
    trainer = training.Trainer(mtx,0,double_precision=True)
    trainer.load_snapshot(snap)

    #####################
    logger.info('initialization test')
    U2,e,alpha2=sp.sparse.linalg.svds(4*(mtx.todense()-.5),Nk)
    U2=U2@np.diag(np.sqrt(e))
    alpha2=alpha2.T@np.diag(np.sqrt(e))
    logits = U2 @ alpha2.T

    tf_logits=trainer.sp.kp_U.dump()['muhat']@trainer.sp.kp_alpha.dump()['muhat'].T

    U2,e,alpha2=sp.sparse.linalg.svds(4*(mtx.todense()[~heldout_rows][:,~heldout_columns]-.5),Nk)
    U2=U2@np.diag(np.sqrt(e))
    alpha2=alpha2.T@np.diag(np.sqrt(e))
    logits2 = U2 @ alpha2.T

    assert np.allclose(logits2,tf_logits[~heldout_rows][:,~heldout_columns])

    ##########################
    logger.info('binary matrix mult test')
    xi=npr.randn(trainer.data.shape[1],2)
    hiddenmask = heldout_rows[:,None] * heldout_columns[None,:]

    tf_version=trainer.X_train.matmul(tf.convert_to_tensor(xi)).numpy()


    Xtilde = np.array(trainer.data.todense())-.5
    Xtilde[hiddenmask]=0
    np_version = Xtilde @ xi

    assert np.allclose(tf_version,np_version)

    ####################
    logger.info('other binary matrix mult test')
    xi=npr.randn(trainer.data.shape[1],2)
    hiddenmask = heldout_rows[:,None] * heldout_columns[None,:]

    tf_version=trainer.X_test.matmul(tf.convert_to_tensor(xi)).numpy()


    Xtilde = np.array(trainer.data.todense())-.5
    Xtilde[~hiddenmask]=0
    np_version = Xtilde @ xi

    assert np.allclose(tf_version,np_version)

    ################################3
    logger.info('KL U test')
    kp=trainer.sp.kp_U
    ekp = ExplodedKP(kp)
    assert np.allclose(
        big_linear_operators.KP_KL(kp).numpy(),
        ekp.KL()
    )

    #############################3
    logger.info("sigma finding test")
    kp = kalman_parameters.KalmanParameters.from_snapshot(snap['kp_U'],True)
    best_sigsq=big_linear_operators.prior_gaussian_update(kp)[1].numpy()
    eps=.01
    mesh=np.exp(np.r_[-eps:eps:10j])
    rez=np.zeros((10,10,2))
    KL=np.zeros((10,10))
    for i in range(rez.shape[0]):
        for j in range(rez.shape[1]):
            rez[i,j] = np.sqrt(best_sigsq*np.r_[mesh[i],mesh[j]])
            kp.update(sigma=tf.convert_to_tensor(rez[i,j]))
            KL[i,j]=big_linear_operators.KP_KL(kp).numpy()
    assert np.unravel_index(np.argmin(KL),KL.shape)==(5,5)

    ###################################
    logger.info('data loss test')
    assert np.allclose(
        data_loss(mtx.todense(),trainer.sp),
        big_linear_operators.data_loss(trainer.X_train,trainer.sp,trainer.rowbatches).numpy()
    )

    #######################################
    logger.info("test loss!")

    tf_version=trainer.loss().heldout_raw_loss*trainer.n_test

    hiddenmask = heldout_rows[:,None] * heldout_columns[None,:]
    Xtilde = np.array(trainer.data.todense())-.5
    Xtilde[~hiddenmask]=0
    U2=trainer.sp.kp_U.muhat.numpy()
    alpha2=trainer.sp.kp_alpha.muhat.numpy()
    logits=U2 @ alpha2.T
    T1=-np.sum(logits*Xtilde)
    T2=np.sum(hiddenmask*np.log(2*np.cosh(logits/2)))

    assert np.allclose(T1+T2,tf_version)

    ###################################
    # rho test

    logger.info("rho test")
    for k in range(Nk):
        
        kp = kalman_parameters.KalmanParameters.from_snapshot(snap['kp_U'],True)
        L1=big_linear_operators.KP_KL(kp).numpy()
        current_rho=kp.rho.numpy()
        L1_mk = big_linear_operators.evaluate_rho_loss(kp,k,current_rho[k]).numpy()
        
        current_rho[k]*=np.exp(npr.randn())
        L2_mk = big_linear_operators.evaluate_rho_loss(kp,k,current_rho[k]).numpy()
        
        kp.update(rho=tf.convert_to_tensor(current_rho))
        L2 = big_linear_operators.KP_KL(kp).numpy()
        
        assert np.allclose(L1-L2,L1_mk-L2_mk)

    ############################
    logger.info('psi mult test')

    xi=npr.randn(Nr,Nk)
    tf_answer=big_linear_operators.mult_Psi(
        trainer.sp,trainer.rowbatches,tf.convert_to_tensor(xi)).numpy()

    G=get_G(trainer.sp)
    EY=np.tanh(G/2)/(2*G)
    M=1-1.0*np.outer(heldout_rows,heldout_columns)


    alphaprep1 = np.array([np.outer(x,x) for x in trainer.sp.kp_alpha.muhat.numpy()])
    alphaprep2 = np.array([np.diag(x**2) for x in trainer.sp.kp_alpha.sighat.numpy()])
    alphaprep=alphaprep1+alphaprep2

    out=[]
    for r in range(Nr):
        v=0
        for c in range(Nc):
            v+= M[r,c]*EY[r,c]*alphaprep[c] @ xi[r]
        out.append(v)
    out=np.array(out)

    assert np.allclose(out,tf_answer)

    ###################################
    logger.info('inversion tests')
    xi=npr.randn(Nr,Nk)
    ppi_xi=big_linear_operators.solve_Psi_Phi(trainer.sp,trainer.rowbatches,tf.convert_to_tensor(xi))[0]

    T1=big_linear_operators.mult_Psi(trainer.sp,trainer.rowbatches,ppi_xi)
    T2=big_linear_operators.mult_Phi(trainer.sp.kp_U,ppi_xi)
    pp_ppi_xi=(T1+T2).numpy()
    assert np.allclose(xi,pp_ppi_xi)

    T1=big_linear_operators.mult_Psi(trainer.sp,trainer.rowbatches,tf.convert_to_tensor(xi))
    T2=big_linear_operators.mult_Phi(trainer.sp.kp_U,tf.convert_to_tensor(xi))
    pp_xi = T1+T2
    ppi_pp_xi=big_linear_operators.solve_Psi_Phi(trainer.sp,trainer.rowbatches,pp_xi)[0].numpy()
    assert np.allclose(xi,ppi_pp_xi)