import cython
import numpy as np

from cython cimport floating, integral

from cython.parallel import parallel, prange

cimport scipy.linalg.cython_blas as cython_blas

# requires scipy v0.16
cimport scipy.linalg.cython_lapack as cython_lapack
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset


# lapack/blas wrappers for cython fused types
cdef inline void axpy(int * n, floating * da, floating * dx, int * incx, floating * dy,
                      int * incy) nogil:
    if floating is double:
        cython_blas.daxpy(n, da, dx, incx, dy, incy)
    else:
        cython_blas.saxpy(n, da, dx, incx, dy, incy)
     
    
cdef inline void symv(char *uplo, int *n, floating *alpha, floating *a, int *lda, floating *x,
                      int *incx, floating *beta, floating *y, int *incy) nogil:
    if floating is double:
        cython_blas.dsymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    else:
        cython_blas.ssymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
        
        
cdef inline floating dot(int *n, floating *sx, int *incx, floating *sy, int *incy) nogil:
    if floating is double:
        return cython_blas.ddot(n, sx, incx, sy, incy)
    else:
        return cython_blas.sdot(n, sx, incx, sy, incy)
    
    
@cython.cdivision(True)
@cython.boundscheck(False)
def calculate_loss(Cui, integral[:] indptr, integral[:] indices, float[:] data,
                floating[:, :] X, floating[:, :] Y, float regularization,
                int num_threads=0):
    dtype = np.float64 if floating is double else np.float32
    cdef integral users = X.shape[0], items = Y.shape[0], u, i, index
    cdef int one = 1, N = X.shape[1]
    cdef floating confidence, temp
    cdef floating zero = 0.

    cdef floating[:, :] YtY = np.dot(np.transpose(Y), Y)

    cdef floating * r

    cdef double loss = 0, total_confidence = 0, item_norm = 0, user_norm = 0

    # Calculate loss = SUM(u,i)[Cui(u,i)(Pui(u,i) - X(u)Y(i))^2] + regularization*(user_norm^2 + item_norm^2)
    with nogil, parallel(num_threads=num_threads):
        r = <floating *> malloc(sizeof(floating) * N)
        try:
            for u in prange(users, schedule='guided'):
                # calculates (A.dot(Xu) - 2 * b).dot(Xu), without calculating A
                temp = 1.0
                #   (UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY )
                # calculates r = YtY * Xu
                symv("U", &N, &temp, &YtY[0, 0], &N, &X[u, 0], &one, &zero, r, &one)

                for index in range(indptr[u], indptr[u + 1]):
                    i = indices[index]
                    confidence = data[index]

                    if confidence > 0:
                        temp = -2 * confidence
                    else:
                        temp = 0
                        confidence = -1 * confidence
                    # calculates (-2 * confidence) + (confidence - 1) * YiXu
                    temp = temp + (confidence - 1) * dot(&N, &Y[i, 0], &one, &X[u, 0], &one)
                    # calculates r = [(-2 * confidence) + (confidence - 1) * YiXu]Yi + YtY*Xu
                    axpy(&N, &temp, &Y[i, 0], &one, r, &one)

                    total_confidence += confidence
                    loss += confidence
                # calculates [[(-2 * confidence) + (confidence - 1) * YiXu]Yi + YtY*Xu] * Xu
                # = [(-2 * confidence) + (confidence - 1) * YiXu]YiXu + YtY*XuXu
                # = -2*confidence*YiXu + confidence*YtYXuXu
                loss += dot(&N, r, &one, &X[u, 0], &one)
                user_norm += dot(&N, &X[u, 0], &one, &X[u, 0], &one)

            for i in prange(items, schedule='guided'):
                item_norm += dot(&N, &Y[i, 0], &one, &Y[i, 0], &one)

        finally:
            free(r)

    loss += regularization * (item_norm + user_norm)
    return loss / (total_confidence + Cui.shape[0] * Cui.shape[1] - Cui.nnz)
