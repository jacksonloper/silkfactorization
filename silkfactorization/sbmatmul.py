import tensorflow as tf

def sbmatmul_original(Nc,mat,row,col,bins):
    result=tf.zeros((Nc,tf.shape(mat)[1]),dtype=mat.dtype)
    for i in range(len(bins)-1):
        rows=row[bins[i]:bins[i+1]]
        cols=col[bins[i]:bins[i+1]]
        vals=tf.gather(mat,cols) # <-- nnz x mat.shape[1]
        result=tf.tensor_scatter_nd_add(result,rows[:,None],vals) # <-- Nc x mat.shape[1]
    return result

sbmatmul_float64=tf.function(sbmatmul_original,input_signature=[
    tf.TensorSpec(shape=[], dtype=tf.int64), # Nc
    tf.TensorSpec(shape=[None,None], dtype=tf.float64), # Mat
    tf.TensorSpec(shape=[None], dtype=tf.int64), # row
    tf.TensorSpec(shape=[None], dtype=tf.int64),# col
    tf.TensorSpec(shape=[None], dtype=tf.int64),# bins
])

sbmatmul_float32=tf.function(sbmatmul_original,input_signature=[
    tf.TensorSpec(shape=[], dtype=tf.int64), # Nc
    tf.TensorSpec(shape=[None,None], dtype=tf.float32), # Mat
    tf.TensorSpec(shape=[None], dtype=tf.int64), # row
    tf.TensorSpec(shape=[None], dtype=tf.int64),# col
    tf.TensorSpec(shape=[None], dtype=tf.int64),# bins
])

sbmatmul_int64=tf.function(sbmatmul_original,input_signature=[
    tf.TensorSpec(shape=[], dtype=tf.int64), # Nc
    tf.TensorSpec(shape=[None,None], dtype=tf.int64), # Mat
    tf.TensorSpec(shape=[None], dtype=tf.int64), # row
    tf.TensorSpec(shape=[None], dtype=tf.int64),# col
    tf.TensorSpec(shape=[None], dtype=tf.int64),# bins
])

def sbmatmul_dispatch(Nc,mat,row,col,bins):
    if mat.dtype==tf.float32:
        return sbmatmul_float32(Nc,mat,row,col,bins)
    elif mat.dtype==tf.float64:
        return sbmatmul_float64(Nc,mat,row,col,bins)
    elif mat.dtype==tf.int64:
        return sbmatmul_int64(Nc,mat,row,col,bins)
    else:
        raise Exception(f"Dtype {mat.dtype} not supported")