#include "concat.h"


concat_layer make_concat_layer(size_params *concat, int lengths){

	concat_layer l ={0};
	l.type = CONCAT;
	int c1,c2,c3;
	if(lengths == 2){
		l.c1 =concat[0].out.c;
		l.c2 =concat[1].out.c;
		l.c = l.c1+l.c2;
	}else {
		l.c1 =concat[0].out.c;
		l.c2 =concat[1].out.c;
		l.c3 =concat[2].out.c;
		l.c = l.c1+l.c2+l.c3;
	}
	l.concat_lengths = lengths;
	l.name = concat[0].top_name;
	l.h = concat[0].out_h;
	l.w = concat[0].out_w;
	l.n = l.c;
	l.batch = concat[0].batch;
	l.out_h = l.h;
	l.out_w = l.w;
	l.out_c = l.n;
	l.inputs = l.w*l.h*l.c;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.input  = calloc(l.batch*l.outputs,sizeof(float));
	l.output = calloc(l.batch*l.outputs,sizeof(float));
	l.delta  = calloc(l.batch*l.outputs,sizeof(float));

#ifdef GPU
	l.forward_gpu = forward_concat_layer_gpu;
	l.backward_gpu = backward_concat_layer_gpu;
	l.updata_gpu = updata_concat_layer_gpu;

#endif
	if(lengths ==2) fprintf(stderr, "concat   %4s + %4s   -> %5s  size: %4d x%4d x%4d\n", n, , concat[0].layer_name, concat[1].layer_name, l.name, l.out_w, l.out_h, l.out_c);
	fprintf(stderr, "concat   %4s + %4s +%4s  -> %5s  size: %4d x%4d x%4d\n", n, , concat[0].layer_name, concat[1].layer_name,concat[2].layer_name, l.name, l.out_w, l.out_h, l.out_c);
	return l;
}