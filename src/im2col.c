#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}
void im2col_cpu_stride1_s3_p1(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = height;
    int width_col = width;
    int sizeRow = width*sizeof(float);

    int channels_col = channels * ksize * ksize;

    for (c = 0; c < channels; c++)
    {
	    float *pDst1 = data_col+c*ksize*ksize*height_col*width_col;
	    float *pSrc = data_im+c*height*width;
	    int offsetCh = height_col*width_col;
	    int offsetToLast = offsetCh - width_col;
	    float *pDst2 = pDst1 + offsetCh;
	    float *pDst3 = pDst2 + offsetCh;
	    float *pDst4 = pDst3 + offsetCh;
	    float *pDst5 = pDst4 + offsetCh;
	    float *pDst6 = pDst5 + offsetCh;
	    float *pDst7 = pDst6 + offsetCh;
	    float *pDst8 = pDst7 + offsetCh;
	    float *pDst9 = pDst8 + offsetCh;
        for (w = 0; w < width_col; ++w)
        {
	    pDst1[w] = 0.0;
	    pDst2[w] = 0.0;
	    pDst3[w] = 0.0;
        }
	
        for (h = 0; h < height_col; h++)
	{
		if (h < height_col-1){		
		pDst1[(h+1)*width_col] = 0.0;
		memcpy(pDst1+(h+1)*width_col+1, pSrc, sizeRow-sizeof(float));
		memcpy(pDst2+(h+1)*width_col, pSrc, sizeRow);
		memcpy(pDst3+(h+1)*width_col, pSrc+1, sizeRow-sizeof(float));
		pDst3[(h+2)*width_col -1] = 0.0;
		}

		pDst4[h*width_col] = 0.0;
		memcpy(pDst4+1+h*width_col, pSrc, sizeRow-sizeof(float));
		memcpy(pDst5+h*width_col, pSrc, sizeRow);
		memcpy(pDst6+h*width_col, pSrc+1, sizeRow-sizeof(float));
		pDst6[(h+1)*width_col -1] = 0.0;
		
		if (h > 0){
		pDst7[(h-1)*width_col] = 0.0;
		memcpy(pDst7+1+(h-1)*width_col, pSrc, sizeRow-sizeof(float));
		memcpy(pDst8+(h-1)*width_col, pSrc, sizeRow);
		memcpy(pDst9+(h-1)*width_col, pSrc+1, sizeRow-sizeof(float));
		pDst9[h*width_col -1] = 0.0;
		}
		pSrc+=width;
        }

        for (w = 0; w < width_col; ++w)
        {
	    pDst7[(h-1)*width_col+w] = 0.0;
	    pDst8[(h-1)*width_col+w] = 0.0;
	    pDst9[(h-1)*width_col+w] = 0.0;
        }
	    

    }
    /*for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }*/
}

