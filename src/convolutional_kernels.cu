#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}
//#define PRUNE_UPDATE 
//#define PRUNE
#define TBLOCK_SIZE 128

inline int compare(const void*a, const void*b)
{
    return *(int *)b - *(int *)a;
}

__global__ void prune_weights_kernel( float* d_weights, const float* __restrict__ d_prune_index, const int N )
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if( tidx < N  ) d_weights[tidx] *= __ldg(d_prune_index+tidx);
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += abs(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += abs(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

//code by lh
//kernel for PRUNE_UPDATE
__global__ void mean_sq_gpu(const float * __restrict__ weights_gpu, float *weights_mean_gpu,float *weights_sq_gpu,const int N,float *d_prune_index)
{
    const int tidx = blockDim.x*blockIdx.x + threadIdx.x;
    __shared__ float SMEM_FABS[128];
    __shared__ float SMEM_SQ[128];
    float weights_per_thread = 0;
    float temp = 0;
    if(tidx<N) {
        weights_per_thread = __ldg(weights_gpu+tidx);  //FOR FABS MEAN
        temp = weights_per_thread * weights_per_thread;//FOR SQ
        d_prune_index[tidx] = 1.0f;
    }
    SMEM_FABS[threadIdx.x] = fabs(weights_per_thread);
    SMEM_SQ[threadIdx.x]   = temp;
    __syncthreads();
    float mySum_FABS = fabs(weights_per_thread);
    float mySum_SQ = temp;
    if(threadIdx.x<64) {
        SMEM_FABS[threadIdx.x] = mySum_FABS = mySum_FABS + SMEM_FABS[threadIdx.x+64];
        SMEM_SQ[threadIdx.x]   = mySum_SQ   = mySum_SQ + SMEM_SQ[threadIdx.x+64];
    }
    __syncthreads();
    if(threadIdx.x<32) {
        SMEM_FABS[threadIdx.x] = mySum_FABS = mySum_FABS + SMEM_FABS[threadIdx.x+32];
        SMEM_SQ[threadIdx.x]   = mySum_SQ   = mySum_SQ + SMEM_SQ[threadIdx.x+32];
    }
    mySum_FABS+=__shfl_down(mySum_FABS,16);
    mySum_SQ  +=__shfl_down(mySum_SQ,16);
    mySum_FABS+=__shfl_down(mySum_FABS,8);
    mySum_SQ  +=__shfl_down(mySum_SQ,8);
    mySum_FABS+=__shfl_down(mySum_FABS,4);
    mySum_SQ  +=__shfl_down(mySum_SQ,4);
    mySum_FABS+=__shfl_down(mySum_FABS,2);
    mySum_SQ  +=__shfl_down(mySum_SQ,2);
    mySum_FABS+=__shfl_down(mySum_FABS,1);
    mySum_SQ  +=__shfl_down(mySum_SQ,1);
    if(threadIdx.x == 0){
        atomicAdd(weights_mean_gpu,mySum_FABS/N);
        atomicAdd(weights_sq_gpu,mySum_SQ);
    }
}

__global__ void forward_prune( float* weights_gpu_tmp, const float * __restrict__ weights_gpu, float* d_prune_index,
                               const int N, const float rate, const float weights_mean_gpu, const float weights_std_gpu )
{
    const int tidx = blockDim.x*blockIdx.x + threadIdx.x;

    if(tidx<N) 
    {
        const float MAX_ = max(weights_mean_gpu+rate*weights_std_gpu,0.f);
        float d_data     = __ldg(weights_gpu + tidx); 
        if(d_prune_index[tidx] == 1.f && d_data <= 0.9f*MAX_){
            d_prune_index[tidx] = 0.f;
            weights_gpu_tmp[tidx] = 0.f;
        }
        else if(d_prune_index[tidx] == 0.f && d_data > 1.1f*MAX_){
            d_prune_index[tidx] = 1.f;
            weights_gpu_tmp[tidx] = d_data;
        }
    }
}

//coded by linhao for darknet prune
void forward_convolutional_layer_gpu(convolutional_layer l, network_state state,int n)
{
    //printf("enter_forward_gpu\n");
    //printf("flag = %d\n", *(l.flag));
    //coded by linhao
// #ifdef PRUNE
//     if(l.size > 1  && *(l.flag)==1)
//     {
//         // open
//         const float ratio = 0.3; //PRUNE 70%
//         const int num_weights = l.n*l.c*l.size*l.size;
//         const int index_threshold = (int)(num_weights * ratio);
//         float *mutable_weights = l.weights;
//         float *fabs_weights = NULL;
//         fabs_weights = (float*)malloc(num_weights*sizeof(float));
//         for(int i = 0;i<num_weights;++i) fabs_weights[i] = fabs(mutable_weights[i]);
//         //sort for weights for each layer, from large to small
//         qsort(fabs_weights, num_weights, sizeof(float), compare);
//         if(index_threshold >= 1){
//             float threshold_weight = fabs_weights[index_threshold - 1];
//             for(int i = 0;i<num_weights;++i){
//                 if(mutable_weights[i]>= threshold_weight || mutable_weights[i] <= -threshold_weight){
//                     (l.h_prune_index)[i]  = 1.f;
//                 }
//                 else{
//                     mutable_weights[i] = 0.f;
//                 }
//             }
//         }
//         else{
//             for(int i =0;i<num_weights;++i) (l.h_prune_index)[i] = (float)1.0f;
//         }
//         check_error(cudaMemcpy(l.d_prune_index, l.h_prune_index, sizeof(float)*num_weights, cudaMemcpyHostToDevice));
//         printf("prune_over\n");
//         if(fabs_weights){
//             free(fabs_weights); fabs_weights = NULL;
//         }
//         cudaMemcpy(l.weights_gpu, l.weights, sizeof(float)*l.size*l.size*l.n*l.c, cudaMemcpyHostToDevice);
//         *(l.flag) = 0;
//     }
// #endif

    //coded by linhao August 22
// #ifdef PRUNE_UPDATE
//     const float rate = 1.8f;  
//     const int weights_num = l.n*l.c*l.size*l.size;
//     //mx: std use managed memory
//     if(l.size == 3 && *(l.flag) == 1){
//         cudaMemcpy(l.weights_gpu_temp, l.weights_gpu,sizeof(float)*weights_num, cudaMemcpyDeviceToDevice);
//         mean_sq_gpu<<<(weights_num+TBLOCK_SIZE-1)/TBLOCK_SIZE,TBLOCK_SIZE>>>(l.weights_gpu,l.weights_mean_gpu,l.weights_sq_gpu,weights_num,l.d_prune_index);
//         //sqrtf(std / N - mu * mu)
//         cudaDeviceSynchronize();
//         *(l.weights_sq_gpu) = sqrtf(*(l.weights_sq_gpu)/weights_num - (*(l.weights_mean_gpu)) * (*(l.weights_mean_gpu)));
//         *(l.flag) = 0;
//     }
//     const float random    = rand() * 1.f / (float)(RAND_MAX);
//     const float threshold = 1.f / (1.f + 0.0001f * (*(l.iters)));
//     if(l.size == 3 /*&& *(l.iters)<l.local_iter*/ && threshold >= random ){
//         forward_prune<<<(weights_num+TBLOCK_SIZE-1)/TBLOCK_SIZE,TBLOCK_SIZE>>>(l.weights_gpu_temp,l.weights_gpu,l.d_prune_index,weights_num,rate,*(l.weights_mean_gpu),*(l.weights_sq_gpu));
//     }
//     (*(l.iters)) += 1;
// #endif
    int bottom_size = l.batch*l.c*l.w*l.h;
    int top_size = l.batch*l.out_c*l.out_w*l.out_h;
    if(l.bottom_name == "data") {
        cudaMemcpy(l.bottom_data_gpu,state.input,sizeof(bottom_size)*float,cudaMemcpyDeviceToDevice);
    }
    else{
        for(int i =0;i<state.net.n;i++){
            if(state.net.layer[i].name == l.bottom_name) cudaMemcpy(l.bottom_data_gpu, state.net.layer[i].top_data, sizeof(top_size), cudaMemcpyDeviceToDevice);
        }
    }
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(l.bottom_data_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        l.bottom_data_gpu = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                l.bottom_data_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                state.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else
    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        im2col_ongpu(l.bottom_data_gpu + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        //coded by linhao August 22
// #ifdef PRUNE_UPDATE
//         float* a = NULL;
//         if(l.size > 1 ){a = l.weights_gpu_temp;}
//         else{a = l.weights_gpu;}
// #endif

        float * a = l.weights_gpu;
        float * b = state.workspace;
        float * c = l.output_gpu;
        gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
    }
#endif

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);

    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
    return;
}

void backward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    //constrain_ongpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, state);
        //axpy_ongpu(l.outputs*l.batch, -state.net.decay, l.x_gpu, 1, l.delta_gpu, 1);
    } else {
        //axpy_ongpu(l.outputs*l.batch, -state.net.decay, l.output_gpu, 1, l.delta_gpu, 1);
    }
    float *original_input = state.input;

    if(l.xnor) state.input = l.binary_input_gpu;

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            state.input,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            state.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(state.delta){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                state.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                state.delta);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
    }

#else
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    for(int i = 0; i < l.batch; ++i)
    {
        float * a = l.delta_gpu;
        float * b = state.workspace;
        float * c = l.weight_updates_gpu;

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(state.delta){
            if(l.binary || l.xnor) swap_binary(&l);
            //float * a = l.weights_gpu;   // weights  
            //coded by linhao August 22
            float* a = NULL;
            if(l.size > 1){a = l.weights_gpu_temp;}
            else{a = l.weights_gpu;}
            float * b = l.delta_gpu;     // top_diff
            float * c = state.workspace; // bottom_dif 

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i*l.c*l.h*l.w);
            if(l.binary || l.xnor) {
                swap_binary(&l);
            }
            if(l.xnor) gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta + i*l.c*l.h*l.w);
        }
    }
//coded by linhao for prune
// #ifdef PRUNE
//     if( 3 == l.size)
//     {
//         const int count = l.n*l.c*l.size*l.size;
//         prune_weights_kernel<<<(count + TBLOCK_SIZE - 1)/TBLOCK_SIZE, TBLOCK_SIZE>>>(l.weight_updates_gpu, l.d_prune_index, count);
//     }    
// #endif

#endif
    return;
}

void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_pull_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_pull_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);

    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_push_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_push_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}

void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;
    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    if(layer.scales_gpu){
        axpy_ongpu(layer.n, learning_rate/batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
        scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);
    }

    if(layer.adam){
        scal_ongpu(size, layer.B1, layer.m_gpu, 1);
        scal_ongpu(size, layer.B2, layer.v_gpu, 1);

        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);

        axpy_ongpu(size, -(1-layer.B1), layer.weight_updates_gpu, 1, layer.m_gpu, 1);
        mul_ongpu(size, layer.weight_updates_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, (1-layer.B2), layer.weight_updates_gpu, 1, layer.v_gpu, 1);

        adam_gpu(size, layer.weights_gpu, layer.m_gpu, layer.v_gpu, layer.B1, layer.B2, learning_rate/batch, layer.eps, layer.t+1);
        fill_ongpu(size, 0, layer.weight_updates_gpu, 1);
    }else{
        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, learning_rate/batch, layer.weight_updates_gpu, 1, layer.weights_gpu, 1);
        scal_ongpu(size, momentum, layer.weight_updates_gpu, 1);
    }
}


