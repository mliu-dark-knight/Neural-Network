from __future__ import division
import numpy as np
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

TILE_SIZE = 32
a_row = 1024
a_col = 256
b_row = 256
b_col = 1024
 

kernel_code_template = """
__global__ void MatrixMultiplication(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
    __shared__ float sA[%(TILE_SIZE)s][%(TILE_SIZE)s];
    __shared__ float sB[%(TILE_SIZE)s][%(TILE_SIZE)s];

    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1) / %(TILE_SIZE)s) + 1); k++) {
        if ( (Row < numARows) && (threadIdx.x + (k*%(TILE_SIZE)s)) < numAColumns)
            sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k * %(TILE_SIZE)s)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;
        if ( Col < numBColumns && (threadIdx.y + k*%(TILE_SIZE)s) < numBRows)
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k * %(TILE_SIZE)s) * numBColumns + Col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();
        for (int j = 0; j < %(TILE_SIZE)s; j++)
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
    }
    if (Row < numCRows && Col < numCColumns)
        C[Row * numCColumns + Col] = Cvalue;
}
"""

# get the kernel code from the template
kernel_code = kernel_code_template % {'TILE_SIZE': TILE_SIZE}

# compile the kernel code
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
MatrixMultiplication = mod.get_function("MatrixMultiplication")

def GPUMatrixMultiplication(a_gpu, b_gpu, c_gpu, a_row, a_col, b_row, b_col):
    MatrixMultiplication(a_gpu, b_gpu, c_gpu, a_row, a_col, b_row, b_col, a_row, b_col, grid = {(b_col / TILE_SIZE) + 1, (a_row / TILE_SIZE) + 1, 1}, block = (TILE_SIZE, TILE_SIZE, 1),)


def test():
    a_cpu = np.random.randn(a_row, a_col).astype(np.float32)
    b_cpu = np.random.randn(b_row, b_col).astype(np.float32)
    c_cpu = np.dot(a_cpu, b_cpu)

    a_gpu = gpuarray.to_gpu(a_cpu) 
    b_gpu = gpuarray.to_gpu(b_cpu)
    c_gpu = gpuarray.empty((a_row, b_col), np.float32)

    GPUMatrixMultiplication(a_gpu, b_gpu, c_gpu, a_row, a_col, b_row, b_col)
    print "-" * 80
    print "CPU-GPU difference:"
    print c_cpu - c_gpu.get()
    print "L2 norm:", la.norm(c_cpu - c_gpu.get())
    np.allclose(c_cpu, c_gpu.get())

