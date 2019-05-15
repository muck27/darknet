#include "darknet.h"
#define K1 1024
float *A;
float *B;
float *C;
float *D;
float *E;
float *F;
float *G;
float *H;
float *I;
float *J;
void cblas_sgemm(int layout, int TransA,
                 int TransB, const int M, const int N,
                 const int K, const float alpha, const float  *A,
                 const int lda, const float  *B, const int ldb,
                 const float beta, float  *C, const int ldc);
extern void gemm(int TA, int TB, int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc);
extern double tot_bn_time;
extern double tot_im2col_time;
extern double tot_act_time;
extern double tot_sgemm_time;
extern double tot_maxpool_time;
extern double tot_fill_time;
extern double tot_lstm_time;
void openblas_set_num_threads(int num_threads);
typedef struct
{
   int m,n,k,i;
   float *D,*E,*F;
}tinfo;
static void *thread_start(void *arg)
{
    tinfo *tinfo = arg;
    int m,n,k,i;
    float *D1,*E1,*F1;
    D1 = tinfo->D; E1 = tinfo->E; F1 = tinfo->F;
    m = tinfo->m; n = tinfo->n; k = tinfo->k;
    for (i = tinfo->i; i < 104; i++)
    {
        /*m = 1; n = 1024; k = 512;
        gemm(0,1,m,n,k,1,A+i*512,k,B,k,1,C+i*1024,n);*/
        m = 1; n = 1024; k = 256;
        gemm(0,1,m,n,k,1,D1+i*256,k,E1,k,1,F1+i*1024,n);
     
    }

}

void test_crnn(char *cfgfile, char *weightfile, char *filename)
{
    int i,j;
    network *net = load_network(cfgfile, weightfile, 0);
    //set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    //float nms=.45;
    //gemm(0,0,M,n,k,1,a,k,b,n,1,c,n);
#if 0
    A = (float *)malloc(K1*K1*sizeof(float));
    B = (float *)malloc(K1*K1*sizeof(float));
    C = (float *)malloc(K1*K1*sizeof(float));
    printf("here\n");
    for (i = 0; i < K1*K1; i++)
    {
        A[i] = 0.0; B[i] = 0.0; C[i] = 0.0;
    }
        time=what_time_is_it_now();
     gemm(0,0,K1,K1,K1,1,A,K1,B,K1,1,C,K1);
        printf("gemm: Predicted in %f seconds.\n", what_time_is_it_now()-time);
        time=what_time_is_it_now();
    cblas_sgemm(101, 111, 111, K1, K1, K1, 1.0, A, K1, B, K1,1.0,C,K1);
        printf("gemm: Predicted in %f seconds.\n", what_time_is_it_now()-time);
#endif
#if 0
    A = (float *)malloc(512*104*sizeof(float));
    B = (float *)malloc(512*1024*sizeof(float));
    C = (float *)malloc(1024*104*sizeof(float));
    D = (float *)malloc(256*104*sizeof(float));
    E = (float *)malloc(256*1024*sizeof(float));
    F = (float *)malloc(1024*104*sizeof(float));
    G = (float *)malloc(512*256*sizeof(float));
    H = (float *)malloc(2048*512*sizeof(float));
    I = (float *)malloc(2048*256*sizeof(float));
    J = (float *)malloc(512*sizeof(float));
    printf("here\n");
    for (i = 0; i < 512*104; i++)
    {
        A[i] = 0.0;
    }
    for (i = 0; i < 1024*512; i++)
    {
        B[i] = 0.0;
    }
    for (i = 0; i < 1024*104; i++)
    {
        C[i] = 0.0;
    }
    for (i = 0; i < 256*104; i++)
    {
        D[i] = 0.0;
    }
    for (i = 0; i < 1024*256; i++)
    {
        E[i] = 0.0;
    }
    for (i = 0; i < 1024*104; i++)
    {
        F[i] = 0.0;
    }
    for (i = 0; i < 512*256; i++)
    {
        G[i] = 0.0;
    }
    for (i = 0; i < 2048*512; i++)
    {
        H[i] = 0.0;
    }
    for (i = 0; i < 2048*256; i++)
    {
        I[i] = 0.0;
    }
    openblas_set_num_threads(1);
    time=what_time_is_it_now();
    pthread_attr_t attr;
    tinfo sinfo;
    pthread_t tid;
    void *res;
    int s = pthread_attr_init(&attr);
    sinfo.m = 1; sinfo.n = 1024; sinfo.k = 512; sinfo.i = 52; sinfo.D = D; sinfo.E = E; sinfo.F = F;
    s = pthread_create(&tid, &attr, &thread_start, &sinfo);
    int m,n,k;
#if 0
    for (i = 0; i < 104; i++)
    {
        /*m = 1; n = 1024; k = 512;
        gemm(0,1,m,n,k,1,A+i*512,k,B,k,1,C+i*1024,n);*/
        m = 1; n = 1024; k = 256;
        gemm(0,1,m,n,k,1,D+i*256,k,E,k,1,F+i*1024,n);
     
    }
#endif
    for (i = 0; i < 52; i++)
    {
        /*m = 1; n = 1024; k = 512;
        gemm(0,1,m,n,k,1,A+i*512,k,B,k,1,C+i*1024,n);*/
        m = 1; n = 1024; k = 256;
        gemm(0,1,m,n,k,1,D+i*256,k,E,k,1,F+i*1024,n);
     
    }
    s = pthread_join(tid, &res);
    openblas_set_num_threads(2);
    m = 52; n = 2048; k = 512;
    gemm(0,1,m,n,k,1,A,k,H,k,1,C,n);

    printf("lstm: compute = %d :: Predicted in %f seconds.\n", 104*1024*768, what_time_is_it_now()-time);
    time=what_time_is_it_now();
    m = 52; n = 256; k = 512;
    gemm(0,1,m,n,k,1,A,k,G,k,1,F,n);
    gemm(0,1,m,n,k,1,A,k,G,k,1,F,n);
    gemm(0,1,m,n,k,1,A,k,G,k,1,F,n);
    gemm(0,1,m,n,k,1,A,k,G,k,1,F,n);
    gemm(0,1,m,n,k,1,A,k,G,k,1,F,n);
    gemm(0,1,m,n,k,1,A,k,G,k,1,F,n);
    gemm(0,1,m,n,k,1,A,k,G,k,1,F,n);
    gemm(0,1,m,n,k,1,A,k,G,k,1,F,n);
    gemm(0,1,m,n,k,1,A,k,G,k,1,F,n);
    gemm(0,1,m,n,k,1,A,k,G,k,1,F,n);
    printf("linear embedding: compute = %d :: Predicted in %f seconds.\n", m*n*k, what_time_is_it_now()-time);
     /*gemm(0,0,K1,K1,K1,1,A,K1,B,K1,1,C,K1);
        printf("gemm: Predicted in %f seconds.\n", what_time_is_it_now()-time);
        time=what_time_is_it_now();
    cblas_sgemm(101, 111, 111, K1, K1, K1, 1.0, A, K1, B, K1,1.0,C,K1);
        printf("gemm: Predicted in %f seconds.\n", what_time_is_it_now()-time);*/
    
    exit(0);
#endif
    openblas_set_num_threads(2);
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
	int rawMode = 0;
	image sized;
	int len = strlen(input);
	float *Y;
	if ((input[len-3] == 'r')&&(input[len-2]=='a')&&(input[len-1]=='w'))
		rawMode=1;
	if ((input[len-3] == 'b')&&(input[len-2]=='i')&&(input[len-1]=='n'))
		rawMode=2;
        //image im = load_image_color(input,0,0);
        //image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);

	// Sankar 26 Sep 2018 : we have no use for larger image, directly resize to net->w, net->h
	if (rawMode==1)
	{
		FILE *fp = fopen(input, "rb");
		if (fp == NULL)
		{
			fprintf(stderr, "cannot open file %s\n", input);
			continue;
		}
		sized = make_image(net->w, net->h, net->c);
		int cnt = net->w*net->h*net->c*sizeof(float);
		if (fread(sized.data, 1, cnt, fp) != cnt)
		{
			fprintf(stderr, "error in reading data from %s\n", input);
		}
		fclose(fp);
	}
        else if (rawMode == 2)
        {
	    Y = malloc(sizeof(float)*256*26);
	    FILE *fp = fopen(input, "rb");
	    fread(Y, 1, 256*26*sizeof(float), fp);
	    fclose(fp);
	}
	else
	{
		sized = load_image(input, net->w, net->h, net->c);
	}

        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        //layer l = net->layers[net->n-1];

	printf("starting inference\n");
        float *X;
	if (rawMode != 2)
	    X = sized.data;
	else
	    X = Y;
	int i1;
	float *out;
	double tot_time = 0;
	for (i1 = 0; i1 < 1; i1++)
	{
        time=what_time_is_it_now();
        out = network_predict(net, X);
        double ptime = what_time_is_it_now()-time;
	printf("%s: Predicted in %f seconds.\n", input, ptime);
	tot_time += ptime;
        }
	printf("%s: avt pred time  in %f seconds.\n", tot_time/i1);
	//printf("TOTAL TIME :: SGEMM = %f :: IM2COL = %f :: BN = %f :: ACT = %f :: MAXPOOL = %f FILL = %f :: LSTM = %f\n", tot_sgemm_time/i1, tot_im2col_time/i1, tot_bn_time/i1, tot_act_time/i1, tot_maxpool_time/i1, tot_fill_time/i1, tot_lstm_time/i1);
        //time=what_time_is_it_now();
        //out = network_predict(net, X);
        //printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        /*time=what_time_is_it_now();
        out = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        time=what_time_is_it_now();
        out = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);*/
	printf("output length %d\n", net->outputs);
	//print_formated_seq_out("OUTPUT", out, net->time_steps, net->outputs);
	char *predRaw = (char *)malloc(net->time_steps);
	int i, j,predLen;
	char alphabet[50];
	strcpy(alphabet, "-0123456789abcdefghijklmnopqrstuvwxyz");
	printf("strting comparison\n");
	fflush(stdout);
	char *predString = (char *)malloc(net->time_steps+1);
	for (i = 0, predLen=0; i < net->time_steps; i++)
	{
		int bestScore = out[i*net->outputs];
		int bestPos = 0;
		for (j = 1; j < net->outputs; j++)
		{
			if (out[i*net->outputs + j] > bestScore)
			{
				bestScore = out[net->outputs*i+j];
				bestPos = j;
			}
				
		}
		predRaw[i] = alphabet[bestPos];
		if ((predRaw[i] != '-') && (!((i > 0) && (predRaw[i - 1] == predRaw[i]))))
                        predString[predLen++] = predRaw[i];
	}
        predRaw[i] = 0;
	predString[predLen] = 0;
	printf("RAW PRED = %s\n", predRaw);
	printf("DEC PRED = %s\n", predString);

#if 0
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions", 0);
#endif

        }
#endif
        //free_image(im);
	if (rawMode != 2)
        free_image(sized);
        if (filename) break;
    }
}

/*
void censor_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = in_s.data;
        network_predict(net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 0, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int left  = b.x-b.w/2.;
                int top   = b.y-b.h/2.;
                censor_image(in, left, top, b.w, b.h);
            }
        }
        show_image(in, base);
        cvWaitKey(10);
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}

void extract_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    int count = 0;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        show_image(in, base);

        int nboxes = 0;
        float *X = in_s.data;
        network_predict(net, X);
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 1, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int size = b.w*in.w > b.h*in.h ? b.w*in.w : b.h*in.h;
                int dx  = b.x*in.w-size/2.;
                int dy  = b.y*in.h-size/2.;
                image bim = crop_image(in, dx, dy, size, size);
                char buff[2048];
                sprintf(buff, "results/extract/%07d", count);
                ++count;
                save_image(bim, buff);
                free_image(bim);
            }
        }
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}
*/

/*
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets)
{
    network_predict_image(net, im);
    layer l = net->layers[net->n-1];
    int nboxes = num_boxes(net);
    fill_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 0, dets);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
}
*/

// runs crnn type network having convolutional layers followed by lstms
// refer to cfg/crnn.cfg for more details
// currently only test mode supported
// IMPORTANT: after convolutional layers, the LSTM needs input to be transposed
// i.e. hxwxc input is really hxw timestep sequence, each element of vector size c
// this transpose achieved using reorg layer with flatten set to 1
void run_crnn(int argc, char **argv)
{
    if(argc < 6){
        fprintf(stderr, "usage: %s %s test [cfg] [weights] [input img]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    char *cfg = argv[3];
    char *weights = argv[4];
    char *filename = argv[5];
    if(0==strcmp(argv[2], "test")) test_crnn(cfg, weights, filename);
    else fprintf(stderr, "only test mode supported\n");
}
