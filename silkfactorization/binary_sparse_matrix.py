from . import sbmatmul
import tensorflow as tf
import scipy as sp
import scipy.sparse
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BinaryMatrixMinusOneHalfJustACorner:
    '''
    Stores matrices where
    - each entry is in {-.5,+.5,0}
    - all the zero-entries lie in a submatrix where MASK ==0
    - all of the non-zero entries lie in three submatrices where MASK==1
    - MASK is defined as the outer product of two boolean vectors
    - the number of +.5 is relatively small (we'll store it sparsely)
    '''

    def __init__(self,Nr,Nc,row,col,heldout_rows,heldout_columns,batchsize=100,filter=True,batches=None):
        '''
        If filter=False, we do NOT assume that pairs row[i],col[i] lie outside the masked
        zone.  This can make bugs.  filter=True is always safe.
        '''
        self.Nr=Nr
        self.Nc=Nc
        self.batchsize=batchsize
        self.shape=(Nr,Nc)
        self.heldout_rows=heldout_rows
        self.heldout_columns=heldout_columns

        if filter:
            assert heldout_rows.dtype==tf.bool
            assert heldout_columns.dtype==tf.bool
            masked = tf.gather(heldout_rows,row)&tf.gather(heldout_columns,col)
            self.row=tf.boolean_mask(row,masked)
            self.col=tf.boolean_mask(col,masked)
        else:
            self.row=row
            self.col=col

        if batches is None:
            self.batches=tf.convert_to_tensor(np.r_[0:len(self.row):batchsize,len(self.row)])
        else:
            self.batches=batches

    def transpose(self):
        return BinaryMatrixMinusOneHalfJustACorner(self.Nc,self.Nr,self.col,self.row,
            self.heldout_columns,self.heldout_rows,batches=self.batches,filter=False)

    def matmul(self,xi):
        A=sbmatmul.sbmatmul_dispatch(self.Nr,xi,self.row,self.col,self.batches) # Nr x Nk

        # there has got to be a better way to do this
        xi2 = tf.boolean_mask(xi,self.heldout_columns)
        b1= tf.zeros(tf.shape(xi)[1],dtype=xi.dtype) # Nk
        b2= tf.reduce_sum(xi2,axis=0) # Nk
        b = tf.where(~self.heldout_rows[:,None],b1[None,:],b2[None,:]) # Nr x Nk

        return A-.5*b

    def matvec(self,vec):
        return self.matmul(vec[:,None])[:,0]


class BinaryMatrixMinusOneHalfWithCornerCutout:
    '''
    Stores matrices where
    - each entry is in {-.5,+.5,0}
    - all the zero-entries lie in a submatrix where MASK ==1
    - all of the non-zero entries lie in three submatrices where MASK==0
    - MASK is defined as the outer product of two boolean vectors
    - the number of +.5 is relatively small (we'll store it sparsely)
    '''

    def __init__(self,Nr,Nc,row,col,heldout_rows,heldout_columns,batchsize=100,filter=True,batches=None):
        '''
        If filter=False, we do NOT assume that pairs row[i],col[i] lie outside the masked
        zone.  This can make bugs.  filter=True is always safe.
        '''
        self.Nr=Nr
        self.Nc=Nc
        self.batchsize=batchsize
        self.shape=(Nr,Nc)
        self.heldout_rows=heldout_rows
        self.heldout_columns=heldout_columns

        if filter:
            assert heldout_rows.dtype==tf.bool
            assert heldout_columns.dtype==tf.bool
            masked = tf.gather(heldout_rows,row)&tf.gather(heldout_columns,col)
            self.row=tf.boolean_mask(row,~masked)
            self.col=tf.boolean_mask(col,~masked)
        else:
            self.row=row
            self.col=col

        if batches is None:
            self.batches=tf.convert_to_tensor(np.r_[0:len(self.row):batchsize,len(self.row)])
        else:
            self.batches=batches

    def transpose(self):
        return BinaryMatrixMinusOneHalfWithCornerCutout(self.Nc,self.Nr,self.col,self.row,
            self.heldout_columns,self.heldout_rows,batches=self.batches,filter=False)

    def matmul(self,xi):
        A=sbmatmul.sbmatmul_dispatch(self.Nr,xi,self.row,self.col,self.batches) # Nr x Nk

        # there has got to be a better way to do this
        xi1 = tf.boolean_mask(xi,self.heldout_columns)
        xi2 = tf.boolean_mask(xi,~self.heldout_columns)
        b1= tf.reduce_sum(xi1,axis=0) # Nk
        b2= tf.reduce_sum(xi2,axis=0) # Nk
        b = tf.where(~self.heldout_rows[:,None],b1[None,:]+b2[None,:],b2[None,:]) # Nr x Nk

        return A-.5*b

    def matvec(self,vec):
        return self.matmul(vec[:,None])[:,0]

def rebatch_matrix(mtx,batchsize):
    batches=tf.convert_to_tensor(np.r_[0:len(mtx.row):batchsize,len(row)])
    return BinaryMatrixMinusOneHalf(mtx.Nr,mtx.Nc,mtx.row,mtx.col,batches)

def construct_Xm12s_from_sparse_scipy_matrix(mtx,MU,Malpha,batchsize):
    assert sp.sparse.issparse(mtx),'scipy sparse matrix expected'
    Nr,Nc=mtx.shape

    logger.info("converting to coo")
    mtx=mtx.tocoo()

    logger.info("filtering the training data")
    X_train = BinaryMatrixMinusOneHalfWithCornerCutout(
        Nr,Nc,
        tf.convert_to_tensor(mtx.row,dtype=tf.int64),
        tf.convert_to_tensor(mtx.col,dtype=tf.int64),
        tf.convert_to_tensor(MU,dtype=tf.bool),
        tf.convert_to_tensor(Malpha,dtype=tf.bool),
        batchsize,
        filter=True
    ) 

    logger.info("filtering the test data")
    X_test = BinaryMatrixMinusOneHalfJustACorner(
        Nr,Nc,
        tf.convert_to_tensor(mtx.row,dtype=tf.int64),
        tf.convert_to_tensor(mtx.col,dtype=tf.int64),
        tf.convert_to_tensor(MU,dtype=tf.bool),
        tf.convert_to_tensor(Malpha,dtype=tf.bool),
        batchsize,
        filter=True
    ) 

    return X_train,X_test