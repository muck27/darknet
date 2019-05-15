#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int sizex, int sizey, int stridex, int stridey, int paddingx, int paddingy, int yolotype)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = (paddingx > paddingy)?paddingx:paddingy;
    l.yolotype = yolotype;
    // support for differential padding
    l.padx = paddingx;
    l.pady = paddingy;
    l.out_w = (w + 2*paddingx - sizex)/stridex + 1;
    l.out_h = (h + 2*paddingy - sizey)/stridey + 1;
    if (l.yolotype == 1){
	l.out_w = (w + 2*paddingx)/stridex;
	l.out_h = (h + 2*paddingy)/stridey;
    }
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = (sizex>sizey)?sizex:sizey;
    l.stride = (stridex>stridey)?stridex:stridey;
    // support for differential stride and kernelsize
    l.sizex = sizex;
    l.sizey = sizey;
    l.stridex = stridex;
    l.stridey = stridey;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max          %d x %d / %d x %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", sizex, sizey, stridex, stridey, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    // modification to support different x y prms
    l->out_w = (w + l->padx - l->sizex)/l->stridex + 1;
    l->out_h = (h + l->pady - l->sizey)/l->stridey + 1;
    if (l->yolotype){
        l->out_w = (w + 2*l->pad)/l->stride;
        l->out_h = (h + 2*l->pad)/l->stride;
    }
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}
extern double maxpool_time;
#define MAX2(a,b) ((a)>(b)?(a):(b))
#define MAX4(a,b,c,d) MAX2(MAX2(a,b),MAX2(c,d))
void forward_maxpool_2x2_s2(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    //printf("mp 2x2\n");
    for(b = 0; b < l.batch*c; ++b)
    {
        float *pDst = l.output + b*h*w;
        float *pSrc1 = net.input + b*4*h*w;
        float *pSrc2 = pSrc1 + 2*w;
	if (l.activation == LEAKY)
        {
	for (i = 0; i < h; i++)
        {
            for (j = 0; j < w; j++,pSrc2+=2, pSrc1+=2, pDst++)
            {
		float a = MAX4(pSrc1[0], pSrc1[1], pSrc2[0], pSrc2[1]);
		*pDst = (a>0)?a:0.1*a;
            }
	    pSrc1+=2*w; pSrc2+=2*w;
        }
	}
	else
	{
	for (i = 0; i < h; i++)
        {
            for (j = 0; j < w; j++,pSrc2+=2, pSrc1+=2, pDst++)
            {
		float a = MAX4(pSrc1[0], pSrc1[1], pSrc2[0], pSrc2[1]);
		*pDst = a;
            }
	    pSrc1+=2*w; pSrc2+=2*w;
        }
	}
    }
}
#define MAXPOOLOPT
void forward_maxpool_layer(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.padx; // we assume padx or y is on either side
    int h_offset = -l.pady;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    
    double time = what_time_is_it_now();
#ifdef MAXPOOLOPT
    if ((l.padx == 0)&&(l.pady==0)&&(l.h%2==0)&&(l.w%2==0)&&(l.sizex == 2)&&(l.sizey==2)&&(l.stridex==2)&&(l.stridey==2))
        forward_maxpool_2x2_s2(l,net);
    else
#endif
    {
    if (l.activation == LEAKY){
    printf("bla\n");
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.sizey; ++n){
                        for(m = 0; m < l.sizex; ++m){
                            int cur_h = h_offset + i*l.stridey + n; // support for diff x and y prm
                            int cur_w = w_offset + j*l.stridex + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    if (max < 0) max *= 0.1;
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
    }
    else
    {
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.sizey; ++n){
                        for(m = 0; m < l.sizex; ++m){
                            int cur_h = h_offset + i*l.stridey + n; // support for diff x and y prm
                            int cur_w = w_offset + j*l.stridex + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
    }
    }
    maxpool_time += what_time_is_it_now()-time;
}

void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}

