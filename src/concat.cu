#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"


#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
	i < (n); \
	i += blockDim.x * gridDim.x)


__global__ void Concat(const int nthreads, const float * __restrict__ in_data,
    const int forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, float* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +(concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = __ldg(in_data+index]);
    } else {
      out_data[index] = __ldg(in_data+top_index);
    }
  }
}


void forward_concat_layer_gpu(layer l,network net,int layers_before){
	fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
	const int concat_input_size_ = l.w*l.h;
	const int num_concats_ = l.batch;
	int offset_concat_axis = 0;
	int num = 0;
	const int top_channels = l.c;
	const int forward = 1;
	for(int j = 1;j<layers_before;j++){
		if(net.layers[j].top_name == l.name){
			const float *bottom_data = net.layers[j].output_gpu;
			const int bottom_concat_axis = net.layers[j].out_c;
			const int bottom_concat_size = bottom_concat_axis*concat_input_size_;
			const int nthreads = bottom_concat_size * num_concats_;
			Concat<<<nthreads/256,256>>>
			    (nthreads,bottom_data,forward,num_concats_,concat_input_size_,
			    top_concat_axis,bottom_concat_axis,offset_concat_axis,l.output_gpu);

			offset_concat_axis += bottom_concat_axis;
			num++;
			if(num==l.concat_lengths) break;
		}
	}
}

void backward_concat_layer_gpu(layer l,network net,int layers_before){





	
}