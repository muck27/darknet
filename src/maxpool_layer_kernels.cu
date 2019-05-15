#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "maxpool_layer.h"
#include "cuda.h"
}
#define MAX2(a,b) ((a)>(b)?(a):(b))
#define MAX4(a,b,c,d) MAX2(MAX2(a,b),MAX2(c,d))

// custom kernel for stride 2, 2x2 kernel, 
__global__ void forward_maxpool_2x2_s2_kernel(int n, int in_h, int in_w, int in_c, int stridex, int stridey, int sizex, int sizey, int padx, int pady, float *input, float *output, int *indexes, int yolotype, int is_leaky, float slope)
{
    int h = (in_h + 2*pady - sizey)/stridey + 1;
    int w = (in_w + 2*padx - sizex)/stridex + 1;
    if (yolotype)
    {
        h = (in_h + 2*pady)/stridey;
        w = (in_w + 2*padx)/stridex;
    }
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -padx;
    int h_offset = -pady;

    int out_index = j + w*(i + h*(k + c*b));
    float *pSrc= input + 2*j + in_w*(2*i+in_h*(k+b*in_c));
    float a = MAX4(pSrc[0], pSrc[1], pSrc[in_w], pSrc[in_w+1]);
    float *pOut = output + j + w*(i + h*(k + c*b));

    if (!is_leaky) 
        pOut[0] = a;
    else
        pOut[0] = ((a > 0)?a:(slope*a));

}









__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stridex, int stridey, int sizex, int sizey, int padx, int pady, float *input, float *output, int *indexes, int yolotype, int is_leaky, float slope)
{
    int h = (in_h + 2*pady - sizey)/stridey + 1;
    int w = (in_w + 2*padx - sizex)/stridex + 1;
    if (yolotype)
    {
        h = (in_h + 2*pady)/stridey;
        w = (in_w + 2*padx)/stridex;
    }
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -padx;
    int h_offset = -pady;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for(l = 0; l < sizey; ++l){
        for(m = 0; m < sizex; ++m){
            int cur_h = h_offset + i*stridey + l;
            int cur_w = w_offset + j*stridex + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    if (is_leaky){ max = (max > 0)? max:(slope*max);}
    output[out_index] = max;
    indexes[out_index] = max_i;
}

__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stridex, int stridey, int sizex, int sizey, int padx, int pady, float *delta, float *prev_delta, int *indexes)
{
    int h = (in_h + pady - sizey)/stridey + 1;
    int w = (in_w + padx - sizex)/stridex + 1;
    int c = in_c;
    int areay = (sizey-1)/stridey;
    int areax = (sizex-1)/stridex;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = -padx;
    int h_offset = -pady;

    float d = 0;
    int l, m;
    for(l = -areay; l < areay+1; ++l){
        for(m = -areax; m < areax+1; ++m){
            int out_w = (j-w_offset)/stridex + m;
            int out_h = (i-h_offset)/stridey + l;
            int out_index = out_w + w*(out_h + h*(k + c*b));
            int valid = (out_w >= 0 && out_w < w &&
                     out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    }
    prev_delta[index] += d;
}
#define MAXPOOLOPT
extern "C" void forward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
    int h = layer.out_h;
    int w = layer.out_w;
    int c = layer.c;
    printf("output h = %d  :: w = %d :: c = %d, leaky = %d\n", h,w,c, layer.activation == LEAKY);
    size_t n = h*w*c*layer.batch;
    // special case, even width and height, 2x2 kernel, and no padding. we can split input into 2x2 tiles
    // valid for most initial layers of tiny yolo or yolo.
#ifdef MAXPOOLOPT
    if ((layer.padx == 0)&&(layer.pady==0)&&(layer.h%2==0)&&(layer.w%2==0)&&(layer.sizex == 2)&&(layer.sizey==2)&&(layer.stridex==2)&&(layer.stridey==2))
    {
    	forward_maxpool_2x2_s2_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stridex, layer.stridey, layer.sizex, layer.sizey, layer.padx, layer.pady, net.input_gpu, layer.output_gpu, layer.indexes_gpu, layer.yolotype, layer.activation==LEAKY, 0.1);
    }
    else
#endif
    {
    	forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stridex, layer.stridey, layer.sizex, layer.sizey, layer.padx, layer.pady, net.input_gpu, layer.output_gpu, layer.indexes_gpu, layer.yolotype, layer.activation==LEAKY, 0.1);
    }
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
    size_t n = layer.h*layer.w*layer.c*layer.batch;

    backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stridex, layer.stridey, layer.sizex, layer.sizey, layer.padx, layer.pady, layer.delta_gpu, net.delta_gpu, layer.indexes_gpu);
    check_error(cudaPeekAtLastError());
}

