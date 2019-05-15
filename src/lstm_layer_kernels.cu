#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "maxpool_layer.h"
#include "cuda.h"
}
__device__ float logistic1_activate_kernel(float x){return 1.f/(1.f + expf(-x));}
__device__ float tanh1_activate_kernel(float x){return (2.f/(1 + expf(-2*x)) - 1);}

__global__ void lstm_fwd_gpu_kernel(int n, int nOut, int batch, float *wfigo, float *ufigo, float *c, float *h, float *cell, float *out)
{

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % nOut;
    int b = id / nOut;
    float *w = wfigo + b*4*nOut;
    float *u = ufigo + b*4*nOut;
    float f = logistic1_activate_kernel(w[j]+u[j]);
    float i = logistic1_activate_kernel(w[j+nOut]+u[j+nOut]);
    float g = tanh1_activate_kernel(w[j+2*nOut]+u[j+2*nOut]);
    float o = logistic1_activate_kernel(w[j+3*nOut]+u[j+3*nOut]);
    c[id] = c[id]*f + g*i;
    h[id] = o*tanh1_activate_kernel(c[id]);
    out[id] = h[id];
    cell[id] = c[id];
}


extern "C" void lstm_fwd_gpu(int nOut, int batch, float *wfigo, float *ufigo, float *c, float *h, float *cell, float *out)
{
    size_t n = nOut * batch;

    lstm_fwd_gpu_kernel<<<cuda_gridsize(n), BLOCK>>>(n, nOut, batch, wfigo, ufigo, c,h,cell,out);
    //lstm_fwd_gpu_kernel<<<cuda_gridsize1(n), 32>>>(n, nOut, batch, wfigo, ufigo, c,h,cell,out);
    //lstm_fwd_gpu_kernel<<<cuda_gridsize(n), BLOCK>>>(n, nOut);
    check_error(cudaPeekAtLastError());
}



