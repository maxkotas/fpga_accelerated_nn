#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t qconv1_input[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer12_out[N_LAYER_10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=qconv1_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=qconv1_input,layer12_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight2_t, 144>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 16>(b2, "b2.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale13_t, 16>(s13, "s13.txt");
        nnet::load_weights_from_txt<bias13_t, 16>(b13, "b13.txt");
        nnet::load_weights_from_txt<weight7_t, 173056>(w7, "w7.txt");
        nnet::load_weights_from_txt<bias7_t, 64>(b7, "b7.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale14_t, 64>(s14, "s14.txt");
        nnet::load_weights_from_txt<bias14_t, 64>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight10_t, 640>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 10>(b10, "b10.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale15_t, 10>(s15, "s15.txt");
        nnet::load_weights_from_txt<bias15_t, 10>(b15, "b15.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    qconv1_result_t layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::conv_2d_cl<input_t, qconv1_result_t, config2>(qconv1_input, layer2_out, w2, b2); // qconv1

    qconv1_alpha_result_t layer13_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::normalize<qconv1_result_t, qconv1_alpha_result_t, config13>(layer2_out, layer13_out, s13, b13); // qconv1_alpha

    layer4_t layer4_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::relu<qconv1_alpha_result_t, layer4_t, relu_config4>(layer13_out, layer4_out); // qrelu1

    layer5_t layer5_out[OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // pool1

    auto& layer6_out = layer5_out;
    qdense1_result_t layer7_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::dense<layer5_t, qdense1_result_t, config7>(layer6_out, layer7_out, w7, b7); // qdense1

    qdense1_alpha_result_t layer14_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::normalize<qdense1_result_t, qdense1_alpha_result_t, config14>(layer7_out, layer14_out, s14, b14); // qdense1_alpha

    layer9_t layer9_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<qdense1_alpha_result_t, layer9_t, relu_config9>(layer14_out, layer9_out); // qrelu2

    qdense2_result_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer9_t, qdense2_result_t, config10>(layer9_out, layer10_out, w10, b10); // qdense2

    qdense2_alpha_result_t layer15_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::normalize<qdense2_result_t, qdense2_alpha_result_t, config15>(layer10_out, layer15_out, s15, b15); // qdense2_alpha

    nnet::softmax<qdense2_alpha_result_t, result_t, softmax_config12>(layer15_out, layer12_out); // softmax

}

