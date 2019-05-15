#include "lstm_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
}
// Sankar Sep 11, adding bidirect flag to prms
layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int bidirect, int adam)
{
    fprintf(stderr, "LSTM Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l = { 0 };
    l.batch = batch;
    l.type = LSTM;
    l.steps = steps;
    l.inputs = inputs;
    l.bidirect = bidirect;
    int out_scf = 1;
    if (bidirect) out_scf = 2;

    l.uf = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uf) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uf->batch = batch;

    l.ui = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ui) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.ui->batch = batch;

    l.ug = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ug) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.ug->batch = batch;

    l.uo = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uo) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uo->batch = batch;

    /* Sankar Oct 4: allocating a composite layer, to accomodate all 4 gates, f,i,g,o*/
    l.ufigo = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ufigo) = make_connected_layer(batch*steps, inputs, 4*outputs, LINEAR, batch_normalize, adam);
    l.ufigo->batch = batch;

    l.wf = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wf) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wf->batch = batch;

    l.wi = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wi) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wi->batch = batch;

    l.wg = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wg) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wg->batch = batch;

    l.wo = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wo) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wo->batch = batch;
    
    /* Sankar Oct 4: allocating a composite layer, to accomodate all 4 gates, f,i,g,o*/
    l.wfigo = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wfigo) = make_connected_layer(batch*steps, outputs, 4*outputs, LINEAR, batch_normalize, adam);
    l.wfigo->batch = batch;
    if (bidirect)
    {
	    l.uf_r = malloc(sizeof(layer));
	    fprintf(stderr, "\t\t");
	    *(l.uf_r) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	    l.uf_r->batch = batch;

	    l.ui_r = malloc(sizeof(layer));
	    fprintf(stderr, "\t\t");
	    *(l.ui_r) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	    l.ui_r->batch = batch;

	    l.ug_r = malloc(sizeof(layer));
	    fprintf(stderr, "\t\t");
	    *(l.ug_r) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	    l.ug_r->batch = batch;

	    l.uo_r = malloc(sizeof(layer));
	    fprintf(stderr, "\t\t");
	    *(l.uo_r) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	    l.uo_r->batch = batch;

	    /* Sankar Oct 4: allocating a composite layer, to accomodate all 4 gates, f,i,g,o*/
	    l.ufigo_r = malloc(sizeof(layer));
	    fprintf(stderr, "\t\t");
	    *(l.ufigo_r) = make_connected_layer(batch*steps, inputs, 4*outputs, LINEAR, batch_normalize, adam);
	    l.ufigo_r->batch = batch;

	    l.wf_r = malloc(sizeof(layer));
	    fprintf(stderr, "\t\t");
	    *(l.wf_r) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	    l.wf_r->batch = batch;

	    l.wi_r = malloc(sizeof(layer));
	    fprintf(stderr, "\t\t");
	    *(l.wi_r) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	    l.wi_r->batch = batch;

	    l.wg_r = malloc(sizeof(layer));
	    fprintf(stderr, "\t\t");
	    *(l.wg_r) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	    l.wg_r->batch = batch;

	    l.wo_r = malloc(sizeof(layer));
	    fprintf(stderr, "\t\t");
	    *(l.wo_r) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	    l.wo_r->batch = batch;
	    /* Sankar Oct 4: allocating a composite layer, to accomodate all 4 gates, f,i,g,o*/
	    l.wfigo_r = malloc(sizeof(layer));
	    fprintf(stderr, "\t\t");
	    *(l.wfigo_r) = make_connected_layer(batch*steps, outputs, 4*outputs, LINEAR, batch_normalize, adam);
	    l.wfigo_r->batch = batch;
    }

    /* Sankar Nov 8 master concat output */
    l.ufigo_fr = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ufigo_fr) = make_connected_layer(batch*steps, inputs, 8*outputs, LINEAR, batch_normalize, adam);
    l.ufigo_fr->batch = batch*steps;
    
    l.batch_normalize = batch_normalize;
    l.outputs = outputs;

    l.output = calloc(outputs*batch*steps*out_scf, sizeof(float));
    l.state = calloc(outputs*batch*out_scf, sizeof(float));

    l.forward = forward_lstm_layer;
    l.update = update_lstm_layer;

    l.prev_state_cpu =  calloc(batch*outputs*out_scf, sizeof(float));
    l.prev_cell_cpu =   calloc(batch*outputs*out_scf, sizeof(float));
    l.cell_cpu =        calloc(batch*outputs*steps*out_scf, sizeof(float));

    l.f_cpu =           calloc(batch*outputs, sizeof(float));
    l.i_cpu =           calloc(batch*outputs, sizeof(float));
    l.g_cpu =           calloc(batch*outputs, sizeof(float));
    l.o_cpu =           calloc(batch*outputs, sizeof(float));
    l.c_cpu =           calloc(batch*outputs*out_scf, sizeof(float));
    l.h_cpu =           calloc(batch*outputs*out_scf, sizeof(float));
    l.temp_cpu =        calloc(batch*outputs, sizeof(float));
    l.temp2_cpu =       calloc(batch*outputs, sizeof(float));
    l.temp3_cpu =       calloc(batch*outputs, sizeof(float));
    l.dc_cpu =          calloc(batch*outputs, sizeof(float));
    l.dh_cpu =          calloc(batch*outputs, sizeof(float));

#ifdef GPU
    l.forward_gpu = forward_lstm_layer_gpu;
    l.backward_gpu = backward_lstm_layer_gpu;
    l.update_gpu = update_lstm_layer_gpu;

    l.output_gpu = cuda_make_array(0, batch*outputs*steps*out_scf);
    l.delta_gpu = cuda_make_array(0, batch*l.outputs*steps*out_scf);

    l.prev_state_gpu = cuda_make_array(0, batch*outputs*out_scf);
    l.prev_cell_gpu = cuda_make_array(0, batch*outputs*out_scf);
    l.cell_gpu = cuda_make_array(0, batch*outputs*steps*out_scf);

    l.f_gpu = cuda_make_array(0, batch*outputs);
    l.i_gpu = cuda_make_array(0, batch*outputs);
    l.g_gpu = cuda_make_array(0, batch*outputs);
    l.o_gpu = cuda_make_array(0, batch*outputs);
    l.c_gpu = cuda_make_array(0, batch*outputs*out_scf);
    l.h_gpu = cuda_make_array(0, batch*outputs*out_scf);
    l.temp_gpu =  cuda_make_array(0, batch*outputs);
    l.temp2_gpu = cuda_make_array(0, batch*outputs);
    l.temp3_gpu = cuda_make_array(0, batch*outputs);
    l.dc_gpu = cuda_make_array(0, batch*outputs);
    l.dh_gpu = cuda_make_array(0, batch*outputs);
#ifdef CUDNN
        cudnnSetTensor4dDescriptor(l.wf->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wf->out_c, l.wf->out_h, l.wf->out_w); 
        cudnnSetTensor4dDescriptor(l.wi->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wi->out_c, l.wi->out_h, l.wi->out_w); 
        cudnnSetTensor4dDescriptor(l.wg->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wg->out_c, l.wg->out_h, l.wg->out_w); 
        cudnnSetTensor4dDescriptor(l.wo->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wo->out_c, l.wo->out_h, l.wo->out_w); 

        cudnnSetTensor4dDescriptor(l.uf->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uf->out_c, l.uf->out_h, l.uf->out_w); 
        cudnnSetTensor4dDescriptor(l.ui->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ui->out_c, l.ui->out_h, l.ui->out_w); 
        cudnnSetTensor4dDescriptor(l.ug->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ug->out_c, l.ug->out_h, l.ug->out_w); 
        cudnnSetTensor4dDescriptor(l.uo->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uo->out_c, l.uo->out_h, l.uo->out_w); 
        
        cudnnSetTensor4dDescriptor(l.wf_r->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wf_r->out_c, l.wf_r->out_h, l.wf_r->out_w); 
        cudnnSetTensor4dDescriptor(l.wi_r->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wi_r->out_c, l.wi_r->out_h, l.wi_r->out_w); 
        cudnnSetTensor4dDescriptor(l.wg_r->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wg_r->out_c, l.wg_r->out_h, l.wg_r->out_w); 
        cudnnSetTensor4dDescriptor(l.wo_r->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wo_r->out_c, l.wo_r->out_h, l.wo_r->out_w); 

        cudnnSetTensor4dDescriptor(l.uf_r->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uf_r->out_c, l.uf_r->out_h, l.uf_r->out_w); 
        cudnnSetTensor4dDescriptor(l.ui_r->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ui_r->out_c, l.ui_r->out_h, l.ui_r->out_w); 
        cudnnSetTensor4dDescriptor(l.ug_r->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ug_r->out_c, l.ug_r->out_h, l.ug_r->out_w); 
        cudnnSetTensor4dDescriptor(l.uo_r->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uo_r->out_c, l.uo_r->out_h, l.uo_r->out_w); 
#endif

#endif

    return l;
}

void update_lstm_layer(layer l, update_args a)
{
    update_connected_layer(*(l.wf), a);
    update_connected_layer(*(l.wi), a);
    update_connected_layer(*(l.wg), a);
    update_connected_layer(*(l.wo), a);
    update_connected_layer(*(l.uf), a);
    update_connected_layer(*(l.ui), a);
    update_connected_layer(*(l.ug), a);
    update_connected_layer(*(l.uo), a);
}
void print_formated_seq_out(const char *pStr, float *pOut, int seqLen, int outSize);
void lstm_fwd_cpu(int nOut, int batch, float *wfigo, float *ufigo, float *c, float *h, float *cell, float *pOut)
{
    int b;
    for (b = 0; b < batch; b++)
    {
	int j;	
	float *wf, *wi, *wg, *wo, *uf, *ui, *ug, *uo;
	wf = wfigo+4*b*nOut;
	wi = wf+nOut;
	wg = wi+nOut;
	wo = wg+nOut;
	uf = ufigo+4*b*nOut;
	ui = uf+nOut;
	ug = ui+nOut;
	uo = ug+nOut;

	for (j = 0; j < nOut; j++)
	{
	    float f = logistic_activate(wf[j]+uf[j]);
	    float i = logistic_activate(wi[j]+ui[j]);
	    float g = tanh_activate(wg[j]+ug[j]);
	    float o = logistic_activate(wo[j]+uo[j]);
	    c[j] = c[j]*f + g*i;
	    h[j] = o*tanh_activate(c[j]);
	    pOut[j] = h[j];
	    cell[j] = c[j];

	}
	cell += nOut;
	h += nOut;
	pOut += nOut;
	c += nOut;
    }
}
extern double bn_time;
extern double im2col_time;
extern double act_time;
extern double sgemm_time;
extern double fill_time;
extern double lstm_time;
typedef struct
{
   //layer l;
   network s;
   int outputs;
   int out_scf;
   float *pCell;
   float *pOutput;
   int batch;
   int steps;
   float *pHidden;
   float *c;
   layer *ufigo_fr;
   layer *wfigo;

}lstm_tinfo_t;
static void *thread_lstm_pass(void *arg)
{
    lstm_tinfo_t *tinfo = arg;
    int i;
    //network s = tinfo->s;
    //layer l = tinfo->l;
    int out_scf = tinfo->out_scf;
    float *pOutput, *pCell;
    
    
    // set up the connected layer of hidden state ->4 gates to point to last step in seq
    layer wfigo = *(tinfo->wfigo);
    layer *ufigo_fr = tinfo->ufigo_fr;
    increment_layer(&wfigo, tinfo->steps-1);
    
    // point to the last output in sequence, the backward portion
    pOutput = tinfo->pOutput + tinfo->outputs*tinfo->batch*(tinfo->steps*out_scf-1);
    pCell = tinfo->pCell + tinfo->outputs*tinfo->batch*(tinfo->steps*out_scf-1);

    for (i = tinfo->steps-1; i >= 0; --i) {
        tinfo->s.input = tinfo->pHidden + tinfo->outputs * tinfo->batch;
        forward_connected_layer(wfigo, tinfo->s);							

	
	lstm_fwd_cpu(tinfo->outputs, tinfo->batch, wfigo.output, ufigo_fr->output+(i*out_scf+1)*4*tinfo->outputs, tinfo->c + tinfo->outputs*tinfo->batch, tinfo->s.input, pCell, pOutput);

	//if (i == l.steps-1) print_formated_seq_out("OUT_T0_BCK", l.output, 1, l.outputs*l.batch);
        pOutput    -= tinfo->outputs*tinfo->batch*out_scf;
        pCell -= tinfo->outputs*tinfo->batch*out_scf;

        increment_layer(&wfigo, -1);

    }
    return NULL;


}
#ifdef GPU
static void *thread_lstm_pass_gpu(void *arg)
{
    lstm_tinfo_t *tinfo = arg;
    int i;
    //network s = tinfo->s;
    //layer l = tinfo->l;
    int out_scf = tinfo->out_scf;
    float *pOutput, *pCell;
    
    
    // set up the connected layer of hidden state ->4 gates to point to last step in seq
    layer wfigo = *(tinfo->wfigo);
    layer *ufigo_fr = tinfo->ufigo_fr;
    increment_layer(&wfigo, tinfo->steps-1);
    
    // point to the last output in sequence, the backward portion
    pOutput = tinfo->pOutput + tinfo->outputs*tinfo->batch*(tinfo->steps*out_scf-1);
    pCell = tinfo->pCell + tinfo->outputs*tinfo->batch*(tinfo->steps*out_scf-1);

    for (i = tinfo->steps-1; i >= 0; --i) {
        tinfo->s.input_gpu = tinfo->pHidden + tinfo->outputs * tinfo->batch;
        forward_connected_layer_gpu(wfigo, tinfo->s);							

	
	lstm_fwd_gpu(tinfo->outputs, tinfo->batch, wfigo.output_gpu, ufigo_fr->output_gpu+(i*out_scf+1)*4*tinfo->outputs, tinfo->c + tinfo->outputs*tinfo->batch, tinfo->s.input_gpu, pCell, pOutput);

	//if (i == l.steps-1) print_formated_seq_out("OUT_T0_BCK", l.output, 1, l.outputs*l.batch);
        pOutput    -= tinfo->outputs*tinfo->batch*out_scf;
        pCell -= tinfo->outputs*tinfo->batch*out_scf;

        increment_layer(&wfigo, -1);

    }
    return NULL;


}
#endif

void forward_lstm_layer(layer l, network state)
{
    network s = { 0 };
    float *pOutBase = l.output;
    s.train = state.train;
    int out_scf = 1;
    if (l.bidirect) out_scf=2;
    int i;
    //layer wf = *(l.wf);
    //layer wi = *(l.wi);
    //layer wg = *(l.wg);
    //layer wo = *(l.wo);
    layer wfigo = *(l.wfigo);

    //layer uf = *(l.uf);
    //layer ui = *(l.ui);
    //layer ug = *(l.ug);
    //layer uo = *(l.uo);
    //layer ufigo = *(l.ufigo);
    layer ufigo_fr = *(l.ufigo_fr);
    double time = what_time_is_it_now();
    fill_cpu(l.outputs*l.batch*out_scf, 0, l.h_cpu, 1);
    fill_cpu(l.outputs*l.batch*out_scf, 0, l.c_cpu, 1);
    /*fill_cpu(l.outputs * l.batch * l.steps, 0, wf.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wi.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wg.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wo.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, uf.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ui.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ug.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, uo.delta, 1);*/
    if (state.train) {
        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
    }
    fill_time += what_time_is_it_now()-time;
    // fwd lstm
    //printf("output of tanh(509.36588) = %f\n", tanh_activate(400));
    s.input = state.input;
    forward_connected_layer(ufigo_fr, s);

    /*****************************************/
    /* spawn a thread for backward dirn lstm */ 
    /*****************************************/

    // setting 1 thread for openblas as we can split work on 2 cores for 2 threads
    openblas_set_num_threads(1);

    time=what_time_is_it_now();

    pthread_attr_t attr;
    lstm_tinfo_t sinfo;
    pthread_t tid;
    void *res;
    int tstatus = pthread_attr_init(&attr);
    //sinfo.m = 1; sinfo.n = 1024; sinfo.k = 512; sinfo.i = 52; sinfo.D = D; sinfo.E = E; sinfo.F = F;
    sinfo.s = s;
    sinfo.outputs = l.outputs;
    sinfo.out_scf = out_scf;
    sinfo.pCell = l.cell_cpu;
    sinfo.pOutput = l.output;
    sinfo.batch = l.batch;
    sinfo.steps = l.steps;
    sinfo.pHidden = l.h_cpu;
    sinfo.c = l.c_cpu;
    sinfo.ufigo_fr = l.ufigo_fr;
    sinfo.wfigo = l.wfigo_r;

    tstatus = pthread_create(&tid, &attr, &thread_lstm_pass, &sinfo);
   
    for (i = 0; i < l.steps; ++i) {
#if 0
        s.input = l.h_cpu;
        forward_connected_layer(wf, s);
        //if (i == 0) print_formated_seq_out("WF_FWD_OUT", wf.output, 1, wf.outputs);
        
        forward_connected_layer(wi, s);							
        forward_connected_layer(wg, s);							
        forward_connected_layer(wo, s);							

        s.input = state.input;
        forward_connected_layer(uf, s);							
        forward_connected_layer(ui, s);							
        forward_connected_layer(ug, s);							
        forward_connected_layer(uo, s);	
#else
        s.input = l.h_cpu;
	forward_connected_layer(wfigo,s);
        //s.input = state.input;
	//forward_connected_layer(ufigo,s);
#endif						
        //if (i == 0) print_formated_seq_out("UG_FWD_OUT", ug.output, 1, ug.outputs);
#if 0
        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);

        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);	
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);	

        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);	
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);	

        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);	
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);	

        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);			
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);		
	//if (i == 0) print_formated_seq_out("g_t", l.g_cpu, 1, l.outputs*l.batch);
        copy_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);		
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);		
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.c_cpu, 1);			
        axpy_cpu(l.outputs*l.batch, 1, l.temp_cpu, 1, l.c_cpu, 1);	

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.h_cpu, 1);			
        activate_array(l.h_cpu, l.outputs*l.batch, TANH);		
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.h_cpu, 1);	

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.cell_cpu, 1);		
        copy_cpu(l.outputs*l.batch, l.h_cpu, 1, l.output, 1);
#else
        time = what_time_is_it_now();
	lstm_fwd_cpu(l.outputs, l.batch, wfigo.output, ufigo_fr.output+i*out_scf*4*l.outputs, l.c_cpu, l.h_cpu, l.cell_cpu, l.output);
        lstm_time += what_time_is_it_now()-time;
#endif
        state.input += l.inputs*l.batch;
        //if (i == 0) print_formated_seq_out("OUT_T0_FWD", l.output, 1, l.outputs*l.batch);
        // in case of bi lstm, the outputs of fwd and backward
        // are concated for each o/p time step. so out_scf  = 2
        l.output    += l.outputs*l.batch*out_scf;
        l.cell_cpu      += l.outputs*l.batch*out_scf;

        /*increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);*/
        increment_layer(&wfigo, 1);

        /*increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);
        increment_layer(&ufigo, 1);*/
    }
#if 0
    wf = *(l.wf_r);
    wi = *(l.wi_r);
    wg = *(l.wg_r);
    wo = *(l.wo_r);
    wfigo = *(l.wfigo_r);

    uf = *(l.uf_r);
    ui = *(l.ui_r);
    ug = *(l.ug_r);
    uo = *(l.uo_r);
    ufigo = *(l.ufigo_r);
    // point to the last input in sequence
    state.input -= l.inputs*l.batch;
    l.output    -= l.outputs*l.batch;
    l.cell_cpu      -= l.outputs*l.batch;
    increment_layer(&wf, l.steps-1);
    increment_layer(&wi, l.steps-1);
    increment_layer(&wg, l.steps-1);
    increment_layer(&wo, l.steps-1);
    increment_layer(&wfigo, l.steps-1);

    increment_layer(&uf, l.steps-1);
    increment_layer(&ui, l.steps-1);
    increment_layer(&ug, l.steps-1);
    increment_layer(&uo, l.steps-1);
    increment_layer(&ufigo, l.steps-1);
    for (i = l.steps-1; i >= 0; --i) {
#if 0
        s.input = l.h_cpu + l.outputs*l.batch;

        forward_connected_layer(wf, s);							
        forward_connected_layer(wi, s);							
        forward_connected_layer(wg, s);							
        forward_connected_layer(wo, s);							

        s.input = state.input;
        forward_connected_layer(uf, s);							
        forward_connected_layer(ui, s);							
        forward_connected_layer(ug, s);							
        forward_connected_layer(uo, s);
#else
        s.input = l.h_cpu + l.outputs*l.batch;
        forward_connected_layer(wfigo, s);							

        //s.input = state.input;
        //forward_connected_layer(ufigo, s);							

#endif							
	
	//if (i == l.steps-1) print_formated_seq_out("WG_REV_OUT", wg.output, 1, wg.outputs);
#if 0
        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);

        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);	
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);	

        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);	
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);	

        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);	
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);	

        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);			
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);		

        copy_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);		
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);		
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.c_cpu+l.outputs*l.batch, 1);			
        axpy_cpu(l.outputs*l.batch, 1, l.temp_cpu, 1, l.c_cpu+l.outputs*l.batch, 1);	

        copy_cpu(l.outputs*l.batch, l.c_cpu+l.outputs*l.batch, 1, l.h_cpu + l.outputs*l.batch, 1);			
        activate_array(l.h_cpu + l.outputs*l.batch, l.outputs*l.batch, TANH);		
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.h_cpu + l.outputs*l.batch, 1);	

        copy_cpu(l.outputs*l.batch, l.c_cpu+l.outputs*l.batch, 1, l.cell_cpu, 1);		
        copy_cpu(l.outputs*l.batch, l.h_cpu+l.outputs*l.batch, 1, l.output, 1);
#else
        time = what_time_is_it_now();
	lstm_fwd_cpu(l.outputs, l.batch, wfigo.output, ufigo_fr.output+(i*out_scf+1)*4*l.outputs, l.c_cpu + l.outputs*l.batch, l.h_cpu + l.outputs*l.batch, l.cell_cpu, l.output);
        lstm_time += what_time_is_it_now()-time;

#endif
        state.input -= l.inputs*l.batch;
	//if (i == l.steps-1) print_formated_seq_out("OUT_T0_BCK", l.output, 1, l.outputs*l.batch);
        l.output    -= l.outputs*l.batch*out_scf;
        l.cell_cpu      -= l.outputs*l.batch*out_scf;

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);
        increment_layer(&wfigo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
        increment_layer(&ufigo, -1);
    }
#endif
    tstatus = pthread_join(tid, &res);        
    openblas_set_num_threads(2);
   
}

void backward_lstm_layer(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);

    state.input += l.inputs*l.batch*(l.steps - 1);
    if (state.delta) state.delta += l.inputs*l.batch*(l.steps - 1);

    l.output += l.outputs*l.batch*(l.steps - 1);
    l.cell_cpu += l.outputs*l.batch*(l.steps - 1);
    l.delta += l.outputs*l.batch*(l.steps - 1);

    for (i = l.steps - 1; i >= 0; --i) {
        if (i != 0) copy_cpu(l.outputs*l.batch, l.cell_cpu - l.outputs*l.batch, 1, l.prev_cell_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.cell_cpu, 1, l.c_cpu, 1);
        if (i != 0) copy_cpu(l.outputs*l.batch, l.output - l.outputs*l.batch, 1, l.prev_state_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.output, 1, l.h_cpu, 1);

        l.dh_cpu = (i == 0) ? 0 : l.delta - l.outputs*l.batch;

        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);			
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);			

        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);			
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);			

        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);			
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);			

        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);			
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);			

        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);			
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);		
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);			
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);		

        copy_cpu(l.outputs*l.batch, l.delta, 1, l.temp3_cpu, 1);		

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);			
        activate_array(l.temp_cpu, l.outputs*l.batch, TANH);			

        copy_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp2_cpu, 1);		
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.temp2_cpu, 1);			

        gradient_array(l.temp_cpu, l.outputs*l.batch, TANH, l.temp2_cpu);
        axpy_cpu(l.outputs*l.batch, 1, l.dc_cpu, 1, l.temp2_cpu, 1);		

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);			
        activate_array(l.temp_cpu, l.outputs*l.batch, TANH);			
        mul_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp_cpu, 1);		
        gradient_array(l.o_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wo.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;															
        backward_connected_layer(wo, s);	

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uo.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(uo, s);									

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);			
        mul_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);				
        gradient_array(l.g_cpu, l.outputs*l.batch, TANH, l.temp_cpu);		
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wg.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;														
        backward_connected_layer(wg, s);	

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ug.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(ug, s);																

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);			
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);				
        gradient_array(l.i_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);	
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wi.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;
        backward_connected_layer(wi, s);						

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ui.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(ui, s);									

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);		
        mul_cpu(l.outputs*l.batch, l.prev_cell_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.f_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wf.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;
        backward_connected_layer(wf, s);						

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uf.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(uf, s);									

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);			
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.temp_cpu, 1);				
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, l.dc_cpu, 1);				

        state.input -= l.inputs*l.batch;
        if (state.delta) state.delta -= l.inputs*l.batch;
        l.output -= l.outputs*l.batch;
        l.cell_cpu -= l.outputs*l.batch;
        l.delta -= l.outputs*l.batch;

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }
}

#ifdef GPU
void update_lstm_layer_gpu(layer l, update_args a)
{
    update_connected_layer_gpu(*(l.wf), a);
    update_connected_layer_gpu(*(l.wi), a);
    update_connected_layer_gpu(*(l.wg), a);
    update_connected_layer_gpu(*(l.wo), a);
    update_connected_layer_gpu(*(l.uf), a);
    update_connected_layer_gpu(*(l.ui), a);
    update_connected_layer_gpu(*(l.ug), a);
    update_connected_layer_gpu(*(l.uo), a);
}



void forward_lstm_layer_gpu(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    int out_scf = 1;
    if (l.bidirect) out_scf=2;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);
    layer wfigo = *(l.wfigo);
    layer ufigo_fr = *(l.ufigo_fr);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);
    layer ufigo = *(l.ufigo);

    /*fill_gpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1);

    fill_gpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);*/
    fill_gpu(l.outputs*l.batch*out_scf, 0, l.h_gpu, 1);
    fill_gpu(l.outputs*l.batch*out_scf, 0, l.c_gpu, 1);

    if (state.train) {
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
    }
    s.input_gpu = state.input_gpu;
    forward_connected_layer_gpu(ufigo_fr, s);
    
    pthread_attr_t attr;
    lstm_tinfo_t sinfo;
    pthread_t tid;
    void *res;
    int tstatus = pthread_attr_init(&attr);
    //sinfo.m = 1; sinfo.n = 1024; sinfo.k = 512; sinfo.i = 52; sinfo.D = D; sinfo.E = E; sinfo.F = F;
    sinfo.s = s;
    sinfo.outputs = l.outputs;
    sinfo.out_scf = out_scf;
    sinfo.pCell = l.cell_gpu;
    sinfo.pOutput = l.output_gpu;
    sinfo.batch = l.batch;
    sinfo.steps = l.steps;
    sinfo.pHidden = l.h_gpu;
    sinfo.c = l.c_gpu;
    sinfo.ufigo_fr = l.ufigo_fr;
    sinfo.wfigo = l.wfigo_r;

    tstatus = pthread_create(&tid, &attr, &thread_lstm_pass_gpu, &sinfo);

    for (i = 0; i < l.steps; ++i) {
#if 0
        s.input_gpu = l.h_gpu;
        forward_connected_layer_gpu(wf, s);							
        forward_connected_layer_gpu(wi, s);							
        forward_connected_layer_gpu(wg, s);							
        forward_connected_layer_gpu(wo, s);							

        s.input_gpu = state.input_gpu;
        forward_connected_layer_gpu(uf, s);							
        forward_connected_layer_gpu(ui, s);							
        forward_connected_layer_gpu(ug, s);							
        forward_connected_layer_gpu(uo, s);							
#else
        s.input_gpu = l.h_gpu;
	forward_connected_layer_gpu(wfigo,s);
        /*s.input_gpu = state.input_gpu;
	forward_connected_layer_gpu(ufigo,s);*/
#endif
#if 0
        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);

        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);	

        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);	

        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);	

        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);		
        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		
        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);			
        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		

        copy_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.c_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, l.temp_gpu, 1, l.c_gpu, 1);	

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.h_gpu, 1);			
        activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);		
        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.h_gpu, 1);	

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.cell_gpu, 1);		
        copy_gpu(l.outputs*l.batch, l.h_gpu, 1, l.output_gpu, 1);
#else
	lstm_fwd_gpu(l.outputs, l.batch, wfigo.output_gpu, ufigo_fr.output_gpu + out_scf*i*4*l.outputs, l.c_gpu, l.h_gpu, l.cell_gpu, l.output_gpu);
#endif
        state.input_gpu += l.inputs*l.batch;
        // in case of bi lstm, the outputs of fwd and backward
        // are concated for each o/p time step. so out_scf  = 2
        l.output_gpu    += l.outputs*l.batch*out_scf;
        l.cell_gpu      += l.outputs*l.batch*out_scf;

        /*increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);

        increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);*/
    }
#if 0
    /*wf = *(l.wf_r);
    wi = *(l.wi_r);
    wg = *(l.wg_r);
    wo = *(l.wo_r);*/
    wfigo = *(l.wfigo_r);

    /*uf = *(l.uf_r);
    ui = *(l.ui_r);
    ug = *(l.ug_r);
    uo = *(l.uo_r);
    ufigo = *(l.ufigo_r);*/
    // point to the last input in sequence
    state.input_gpu -= l.inputs*l.batch;
    l.output_gpu    -= l.outputs*l.batch;
    l.cell_gpu      -= l.outputs*l.batch;
    /*increment_layer(&wf, l.steps-1);
    increment_layer(&wi, l.steps-1);
    increment_layer(&wg, l.steps-1);
    increment_layer(&wo, l.steps-1);*/
    increment_layer(&wfigo, l.steps-1);

/*    increment_layer(&uf, l.steps-1);
    increment_layer(&ui, l.steps-1);
    increment_layer(&ug, l.steps-1);
    increment_layer(&uo, l.steps-1);
    increment_layer(&ufigo, l.steps-1);*/
    for (i = l.steps-1; i >= 0; --i) {
#if 0
        s.input_gpu = l.h_gpu + l.outputs*l.batch;

        forward_connected_layer_gpu(wf, s);							
        forward_connected_layer_gpu(wi, s);							
        forward_connected_layer_gpu(wg, s);							
        forward_connected_layer_gpu(wo, s);							

        s.input_gpu = state.input_gpu;
        forward_connected_layer_gpu(uf, s);							
        forward_connected_layer_gpu(ui, s);							
        forward_connected_layer_gpu(ug, s);							
        forward_connected_layer_gpu(uo, s);							
#else
        s.input_gpu = l.h_gpu + l.outputs*l.batch;
        forward_connected_layer_gpu(wfigo, s);							

        /*s.input_gpu = state.input_gpu;
        forward_connected_layer_gpu(ufigo, s);*/

#endif
#if 0
	//if (i == l.steps-1) print_formated_seq_out("WG_REV_OUT", wg.output, 1, wg.outputs);
        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);

        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);	

        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);	

        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);	

        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);		
        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		
        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);			
        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		

        copy_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.c_gpu+l.outputs*l.batch, 1);			
        axpy_gpu(l.outputs*l.batch, 1, l.temp_gpu, 1, l.c_gpu+l.outputs*l.batch, 1);	

        copy_gpu(l.outputs*l.batch, l.c_gpu+l.outputs*l.batch, 1, l.h_gpu + l.outputs*l.batch, 1);			
        activate_array_gpu(l.h_gpu + l.outputs*l.batch, l.outputs*l.batch, TANH);		
        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.h_gpu + l.outputs*l.batch, 1);	

        copy_gpu(l.outputs*l.batch, l.c_gpu+l.outputs*l.batch, 1, l.cell_gpu, 1);		
        copy_gpu(l.outputs*l.batch, l.h_gpu+l.outputs*l.batch, 1, l.output_gpu, 1);

#else
	lstm_fwd_gpu(l.outputs, l.batch, wfigo.output_gpu, ufigo_fr.output_gpu + (i*out_scf+1)*4*l.outputs, l.c_gpu + l.outputs*l.batch, l.h_gpu + l.outputs*l.batch, l.cell_gpu, l.output_gpu);

#endif
        state.input_gpu -= l.inputs*l.batch;
	//if (i == l.steps-1) print_formated_seq_out("OUT_T0_BCK", l.output, 1, l.outputs*l.batch);
        l.output_gpu    -= l.outputs*l.batch*out_scf;
        l.cell_gpu      -= l.outputs*l.batch*out_scf;
        /*increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);*/
        increment_layer(&wfigo, -1);

        /*increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
        increment_layer(&ufigo, -1);*/
    }
#endif
    // sync back with the thread that launches the backward pass at sequence
    tstatus = pthread_join(tid, &res);
    // sync back with the gpu stream that launches the backward pass at sequence
    cudaDeviceSynchronize();
}

void backward_lstm_layer_gpu(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);

    state.input_gpu += l.inputs*l.batch*(l.steps - 1);
    if (state.delta_gpu) state.delta_gpu += l.inputs*l.batch*(l.steps - 1);

    l.output_gpu += l.outputs*l.batch*(l.steps - 1);
    l.cell_gpu += l.outputs*l.batch*(l.steps - 1);
    l.delta_gpu += l.outputs*l.batch*(l.steps - 1);

    for (i = l.steps - 1; i >= 0; --i) {
        if (i != 0) copy_gpu(l.outputs*l.batch, l.cell_gpu - l.outputs*l.batch, 1, l.prev_cell_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.cell_gpu, 1, l.c_gpu, 1);
        if (i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.prev_state_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.h_gpu, 1);

        l.dh_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);			

        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);			

        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);			

        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);			

        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);			
        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		
        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);			
        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		

        copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, l.temp3_gpu, 1);		

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			
        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);			

        copy_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.temp2_gpu, 1);			

        gradient_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH, l.temp2_gpu);
        axpy_gpu(l.outputs*l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);		

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			
        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);			
        mul_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);		
        gradient_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wo.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;															
        backward_connected_layer_gpu(wo, s);	

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uo.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(uo, s);									

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
        mul_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);				
        gradient_array_gpu(l.g_gpu, l.outputs*l.batch, TANH, l.temp_gpu);		
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wg.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;														
        backward_connected_layer_gpu(wg, s);	

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ug.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(ug, s);																

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);				
        gradient_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);	
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wi.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;
        backward_connected_layer_gpu(wi, s);						

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ui.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(ui, s);									

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wf.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;
        backward_connected_layer_gpu(wf, s);						

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uf.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(uf, s);									

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.temp_gpu, 1);				
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, l.dc_gpu, 1);				

        state.input_gpu -= l.inputs*l.batch;
        if (state.delta_gpu) state.delta_gpu -= l.inputs*l.batch;
        l.output_gpu -= l.outputs*l.batch;
        l.cell_gpu -= l.outputs*l.batch;
        l.delta_gpu -= l.outputs*l.batch;

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }
}
#endif
