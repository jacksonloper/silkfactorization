from . import binary_sparse_matrix
from . import initialization
from . import kalman_parameters
from . import big_linear_operators
import scipy as sp
import scipy.sparse
import tensorflow as tf
import numpy as np
import types

def make_initial_snapshot(data,Nk,ds_U,ds_alpha,heldout_rows,heldout_columns):
    return initialization.initialize_from_scipy_sparse(data,Nk,ds_U,ds_alpha,
        heldout_rows,heldout_columns)

class Trainer:
    def __init__(self,data,GPU_ram,double_precision=False):
        assert sp.sparse.issparse(data),'scipy sparse matrix expected'
        self.data=data
        self.double_precision=bool(double_precision)
        self.Nr,self.Nc=self.data.shape
        self.GPU_ram=float(GPU_ram)

        if self.double_precision:
            self.dtype=tf.float64
        else:
            self.dtype=tf.float32

        if self.double_precision: 
            self._num_floats_we_can_hold = int(1e9 * GPU_ram / 8) 
        else: 
            self._num_floats_we_can_hold = int(1e9 * GPU_ram / 4) 

    def loss(self):
        rez=types.SimpleNamespace(
            KL_U = big_linear_operators.KP_KL(self.sp.kp_U).numpy(),
            KL_alpha = big_linear_operators.KP_KL(self.sp.kp_alpha).numpy(),
            data_loss = big_linear_operators.data_loss(self.X_train,self.sp,self.rowbatches).numpy(),
            raw_loss = big_linear_operators.data_loss(self.X_train,self.sp,self.rowbatches,raw=True).numpy()/self.n_train,
            heldout_raw_loss = big_linear_operators.test_loss(self.X_test,self.sp,self.test_rowbatches).numpy()/self.n_test,
        )

        rez.total_loss = (rez.data_loss + rez.KL_U + rez.KL_alpha) 
        rez.loss = rez.total_loss/ self.n_train

        return rez

    def update_row_rhos(self,factor=2,bins=10,doit=False):
        currho = self.sp.kp_U.rho
        lf = np.log(factor)

        mults = np.exp(np.r_[-lf:lf:1j*bins])
        out=np.zeros(self.Nk)
        for k in range(self.Nk):
            results = [big_linear_operators.evaluate_rho_loss(self.sp.kp_U,k,currho[k]*mult).numpy() for mult in mults]
            out[k] = mults[np.argmin(results)]

        newrho = currho * tf.convert_to_tensor(out,dtype=currho.dtype)

        if doit:
            self.sp.kp_U.update(rho=newrho)
            self.losses.append(self.loss())
            self.losses[-1].update_type="row_rhos"

        return newrho.numpy(),out

    def update_col_rhos(self,factor=2,bins=10,doit=False):
        currho = self.sp.kp_alpha.rho
        lf = np.log(factor)

        mults = np.exp(np.r_[-lf:lf:1j*bins])
        out=np.zeros(self.Nk)
        for k in range(self.Nk):
            results = [big_linear_operators.evaluate_rho_loss(self.sp.kp_alpha,k,currho[k]*mult).numpy() for mult in mults]
            out[k] = mults[np.argmin(results)]

        newrho = currho * tf.convert_to_tensor(out,dtype=currho.dtype)

        if doit:
            self.sp.kp_alpha.update(rho=newrho)
            self.losses.append(self.loss())
            self.losses[-1].update_type="row_rhos"

        return newrho.numpy(),out

    def update_row_prior(self,doit=False):
        newmu,newsigmasq = big_linear_operators.prior_gaussian_update(self.sp.kp_U)
        newsigma=tf.sqrt(newsigmasq)

        if doit:
            self.sp.kp_U.update(mu=newmu,sigma=newsigma)
            self.losses.append(self.loss())
            self.losses[-1].update_type="row_prior"

        return newmu.numpy(),newsigma.numpy()

    def update_col_prior(self,doit=False):
        newmu,newsigmasq = big_linear_operators.prior_gaussian_update(self.sp.kp_alpha)
        newsigma=tf.sqrt(newsigmasq)

        if doit:
            self.sp.kp_alpha.update(mu=newmu,sigma=newsigma)
            self.losses.append(self.loss())
            self.losses[-1].update_type="row_prior"

        return newmu.numpy(),newsigma.numpy()

    def update_row_variational(self,doit=False,damping='optimal'):
        if damping=='optimal':
            pass
        else:
            damping=tf.convert_to_tensor(float(damping),dtype=self.dtype)

        c,muhat,newsigsqi=big_linear_operators.variational_update(self.X_train,self.sp,self.rowbatches)
        newsighat = 1/tf.sqrt(newsigsqi)

        if doit:
            self.sp.kp_U.update(muhat=muhat,sighat=newsighat)
            self.losses.append(self.loss())
            self.losses[-1].update_type="row_variational"
            self.losses[-1].damping_used=damping
            self.losses[-1].damping_result=c.numpy()

        return c.numpy(),muhat.numpy(),newsighat.numpy()

    def update_col_variational(self,doit=False,damping='optimal'):
        if damping=='optimal':
            pass
        else:
            damping=tf.convert_to_tensor(float(damping),dtype=self.dtype)

        c,muhat,newsigsqi=big_linear_operators.variational_update(
            self.X_train.transpose(),self.sp.transpose(),self.colbatches)
        newsighat = 1/tf.sqrt(newsigsqi)

        if doit:
            self.sp.kp_alpha.update(muhat=muhat,sighat=newsighat)
            self.losses.append(self.loss())
            self.losses[-1].update_type="col_variational"
            self.losses[-1].damping_used=damping
            self.losses[-1].damping_result=c.numpy()

        return c.numpy(),muhat.numpy(),newsighat.numpy()

    def load_snapshot(self,snap):
        self.sp = kalman_parameters.SilkParameters.from_snapshot(snap,self.double_precision)
        self.Nk=self.sp.kp_U.Nk
        self._num_loadings_we_can_hold = int(self._num_floats_we_can_hold / (self.Nk+1))

        if hasattr(self,'X_train'):
            del self.X_train
            del self.X_test
        self.X_train,self.X_test = binary_sparse_matrix.construct_Xm12s_from_sparse_scipy_matrix(
            self.data,
            snap['kp_U']['M'],snap['kp_alpha']['M'],
            2+self._num_loadings_we_can_hold//4)

        nh_r= np.sum(snap['kp_U']['M'])
        nh_c= np.sum(snap['kp_alpha']['M'])

        batchsize = 2+self._num_floats_we_can_hold // (8*self.Nc)
        self.rowbatches = tf.convert_to_tensor(np.r_[0:self.Nr:batchsize,self.Nr])
        batchsize = 2+self._num_floats_we_can_hold // (8*self.Nc)
        self.colbatches = tf.convert_to_tensor(np.r_[0:self.Nc:batchsize,self.Nc])
        batchsize = 2+self._num_floats_we_can_hold // (8*nh_c)
        self.test_rowbatches = tf.convert_to_tensor(np.r_[0:nh_r:batchsize,nh_r])

        self.n_test = nh_r*nh_c
        self.n_train = self.Nc*self.Nr - self.n_test

        self.losses=[self.loss()]

    def dump_snapshot(self):
        return dict(kp_U=self.sp.kp_U.dump(),kp_alpha=self.sp.kp_alpha.dump())