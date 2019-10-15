import tensorflow as tf
from . import sbmatmul
import logging

logger = logging.getLogger(__name__)



def iterate_batches(batches):
    for i in range(len(batches)-1):
        yield batches[i],batches[i+1]

def pge_safe(x):
    switch=tf.abs(x)<.00001
    
    A=.25-0.020833333333333332*(x**2)
    B=tf.tanh(x/2)/(2*x)
    return tf.where(switch,A,B)


def mult_Phi(kp,xi):
    '''
    Multiplies by the diagonal part of the inverse covariance

    expects an argument which is a matrix of size length x Nk
    '''

    return kp.diags*xi

def mult_D(kp,xi):
    '''
    Multiplies by the off-diagonal parts of the inverse covariance

    expects an argument which is a matrix of size length x Nk
    '''

    starting = xi[1]*kp.cross[0] # Nk
    innerbits = xi[:-2]*kp.cross[:-1] + xi[2:]*kp.cross[1:] # length-2 times Nk
    ending = xi[-2]*kp.cross[-1] # Nk

    return tf.concat([starting[None,:],innerbits,ending[None,:]],0)

def mult_D_broadcast(kp,xi):
    '''
    xi is an Nk vector, appropriately broadcast
    '''

    starting = xi*kp.cross[0] # Nk
    innerbits = xi[None,:]*kp.cross[:-1] + xi[None,:]*kp.cross[1:] # length-2 times Nk
    ending = xi*kp.cross[-1] # Nk

    return tf.concat([starting[None,:],innerbits,ending[None,:]],0)

def mult_Sigmai(kp,xi):
    return mult_Phi(kp,xi)-mult_D(kp,xi)

def mult_Sigmai_broadcast(kp,xi):
    return mult_Phi(kp,xi[None,:])-mult_D_broadcast(kp,xi)


def localGsq(sp,st,en):
    muhatU = sp.kp_U.muhat
    sighatUsq = sp.kp_U.sighatsq
    muhatalpha = sp.kp_alpha.muhat
    sighatalphasq = sp.kp_alpha.sighatsq

    G1 = (muhatU[st:en] @ tf.transpose(muhatalpha))**2
    G2 = sighatUsq[st:en] @ tf.transpose(sighatalphasq)
    G3 = muhatU[st:en]**2 @ tf.transpose(sighatalphasq)
    G4 = sighatUsq[st:en] @ tf.transpose(muhatalpha**2)

    return G1+G2+G3+G4

def mult_Psi(sp,rowbatches,xi):
    prepalpha = sp.kp_alpha.outer_hats

    muhatU = sp.kp_U.muhat
    sighatUsq = sp.kp_U.sighatsq
    muhatalpha = sp.kp_alpha.muhat
    sighatalphasq = sp.kp_alpha.sighatsq

    out=[]
    for (st,en) in iterate_batches(rowbatches):
      G = tf.sqrt(localGsq(sp,st,en))
      M = pge_safe(G) * (1-sp.kp_U.M_fl[st:en,None]*sp.kp_alpha.M_fl[None,:])

      Psi = tf.einsum('rc,ckl -> rkl',M,prepalpha) # aggregate over columns
      out.append(tf.einsum('rkl,rl->rk',Psi,xi[st:en])) # many independent matrix-multiplies over loadings

    return tf.concat(out,0)


def solve_Psi_Phi(sp,rowbatches,xi):
    prepalpha = sp.kp_alpha.outer_hats

    out=[]
    newsigsqi=[]
    for (st,en) in iterate_batches(rowbatches):
        G = tf.sqrt(localGsq(sp,st,en))
        M = pge_safe(G) * (1-sp.kp_U.M_fl[st:en,None]*sp.kp_alpha.M_fl[None,:])

        Psi_Plus_Phi = tf.einsum('rc,ckl -> rkl',M,prepalpha) + tf.linalg.diag(sp.kp_U.diags[st:en])
        out.append(tf.linalg.solve(Psi_Plus_Phi,xi[st:en][:,:,None])[:,:,0])
        newsigsqi.append(sp.kp_U.diags[st:en] + M@(sp.kp_alpha.sighatsq + sp.kp_alpha.muhat**2))

    return tf.concat(out,0),tf.concat(newsigsqi,0)


def data_loss(X_train,sp,rowbatches,raw=False):
    T1 = tf.reduce_sum(sp.kp_U.muhat * X_train.matmul(sp.kp_alpha.muhat))

    coshes=[]
    for (st,en) in iterate_batches(rowbatches):
        if raw:
            G = sp.kp_U.muhat[st:en] @ tf.transpose(sp.kp_alpha.muhat)
        else:
            G = tf.sqrt(localGsq(sp,st,en))

        # logger.debug(str(G.numpy()))

        # want   log 2 cosh G/2
        #      = log e^-G/2 + e^G/2
        #      = log(1 + e^G) - .5*G
        #      = softplus(G) - .5*G
        sub = (tf.nn.softplus(G)-.5*G)

        # filter out test data
        sub = sub * (1-sp.kp_U.M_fl[st:en,None]*sp.kp_alpha.M_fl[None,:])

        # done for this batch
        coshes.append(tf.reduce_sum(sub))

    logger.debug("data loss = "+str(-T1.numpy()))
    logger.debug("cosh_loss = "+str(tf.reduce_sum(coshes).numpy()))

    return tf.reduce_sum(coshes)-T1 

def test_loss(X_test,sp,rowbatches):
    T1 = tf.reduce_sum(sp.kp_U.muhat * X_test.matmul(sp.kp_alpha.muhat))

    # filter for test data
    U_sub = tf.boolean_mask(sp.kp_U.muhat,sp.kp_U.M,axis=0)
    alpha_sub = tf.boolean_mask(sp.kp_alpha.muhat,sp.kp_alpha.M,axis=0)


    # compute coshes
    coshes=[]
    for (st,en) in iterate_batches(rowbatches):
        G=U_sub[st:en] @ tf.transpose(alpha_sub)
        coshes.append(tf.reduce_sum(tf.nn.softplus(G)-.5*G))

    logger.debug("test_data loss = "+str(-T1.numpy()))
    logger.debug("test_cosh_loss = "+str(tf.reduce_sum(coshes).numpy()))

    return tf.reduce_sum(coshes) -T1 


def variational_update(X_train,sp,rowbatches,c='optimal'):
    b= X_train.matmul(sp.kp_alpha.muhat) + mult_Sigmai_broadcast(sp.kp_U,sp.kp_U.mu)
    xi = b + mult_D(sp.kp_U,sp.kp_U.muhat)
    nu,newsigsqi = solve_Psi_Phi(sp,rowbatches,xi)


    delta = nu - sp.kp_U.muhat
    if c is 'optimal':
        AD = mult_Psi(sp,rowbatches,delta) + mult_Sigmai(sp.kp_U,delta)

        num = tf.reduce_sum(delta*b) - tf.reduce_sum(sp.kp_U.muhat * AD)
        denom = tf.reduce_sum(delta * AD)
        c = num/denom

    return c,sp.kp_U.muhat + c*delta,newsigsqi

def prior_gaussian_update(kp):
    ones = tf.ones((kp.length,kp.Nk),dtype=kp.muhat.dtype)
    sigsums = mult_Sigmai(kp,ones)
    newmu = tf.reduce_sum(kp.muhat * sigsums,axis=0) / tf.reduce_sum(sigsums,axis=0) 

    T1 = tf.reduce_mean(kp.sigsq[None,:]*kp.sighat * mult_Phi(kp,kp.sighat),axis=0)

    diff = (kp.mu[None,:]-kp.muhat)
    T2 = tf.reduce_mean(kp.sigsq[None,:]*diff * mult_Sigmai(kp,diff),axis=0)

    return newmu,T1+T2

def evaluate_rho_loss(kp,k,rho):
    pi = tf.exp(-.5*rho*kp.ds) # length + 1, first and last entry are ZERO
    pisq = tf.exp(-rho*kp.ds)

    diags = kp.sigsqi[k]*(1 - pisq[1:]*pisq[:-1]) / ((1 - pisq[1:]) * (1 - pisq[:-1])) # length x Nk
    cross = kp.sigsqi[k]*pi[1:-1] / (1-pisq[1:-1]) # length - 1

    T1= tf.reduce_sum(kp.sighatsq[:,k] * diags)

    dk = kp.muhat[:,k] - kp.mu[k]
    T2 = tf.reduce_sum(diags* dk**2 ) - 2*tf.reduce_sum(cross*dk[:-1]*dk[1:])

    T3 = tf.reduce_sum(tf.math.log(1-pisq[1:-1]))

    return .5*(T1+T2+T3)

def logdet_Sig(kp):
    T1 = kp.length*tf.reduce_sum(tf.math.log(kp.sigsq))
    T2 = tf.reduce_sum(tf.math.log(1-kp.pisq[1:-1]))
    return T1+T2

def KP_KL(kp):
    trace_term = tf.reduce_sum(kp.sighat*mult_Phi(kp,kp.sighat))

    diff= kp.muhat - kp.mu
    mu_term = tf.reduce_sum(diff * mult_Sigmai(kp,diff))

    constant_term = -kp.length*kp.Nk

    neg_det_Sighat = -tf.reduce_sum(tf.math.log(kp.sighatsq))
    det_Sig = logdet_Sig(kp)

    logger.debug('trace_term = '+str(trace_term.numpy()))
    logger.debug('mu_term = '+str(mu_term.numpy()))
    logger.debug('constant_term = '+str(constant_term))
    logger.debug('neg_det_Sighat = '+str(neg_det_Sighat.numpy()))
    logger.debug('det_Sig = '+str(det_Sig.numpy()))

    return .5*(trace_term+mu_term+constant_term+neg_det_Sighat+det_Sig)