#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"



#ifdef GPU 
void forward_concat_layer_gpu(layer l,network net);
void backward_concat_layer_gpu(layer l,network net);
void updata_concat_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay);
#endif

layer make_concat_layer(size_params *concat, int lengths);
void forward_concat_layer_cpu (layer l,network net){};
void backward_concat_layer_cpu(layer l,network net){};
#endif