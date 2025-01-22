#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 28
#define N_INPUT_2_1 28
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 26
#define OUT_WIDTH_2 26
#define N_FILT_2 16
#define OUT_HEIGHT_2 26
#define OUT_WIDTH_2 26
#define N_FILT_2 16
#define OUT_HEIGHT_2 26
#define OUT_WIDTH_2 26
#define N_FILT_2 16
#define OUT_HEIGHT_5 13
#define OUT_WIDTH_5 13
#define N_FILT_5 16
#define N_SIZE_0_6 2704
#define N_LAYER_7 64
#define N_LAYER_7 64
#define N_LAYER_7 64
#define N_LAYER_10 10
#define N_LAYER_10 10
#define N_LAYER_10 10


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<25,14> qconv1_result_t;
typedef ap_fixed<4,3> weight2_t;
typedef ap_fixed<4,3> bias2_t;
typedef ap_fixed<16,6> qconv1_alpha_result_t;
typedef struct exponent_scale13_t {ap_uint<1> sign;ap_int<4> weight; } exponent_scale13_t;
typedef ap_fixed<4,3> bias13_t;
typedef ap_ufixed<4,0,AP_RND_CONV,AP_SAT,0> layer4_t;
typedef ap_fixed<18,8> qrelu1_table_t;
typedef ap_ufixed<4,0,AP_RND_CONV,AP_SAT,0> layer5_t;
typedef ap_fixed<21,16> qdense1_result_t;
typedef ap_fixed<4,3> weight7_t;
typedef ap_fixed<4,3> bias7_t;
typedef ap_uint<1> layer7_index;
typedef ap_fixed<16,6> qdense1_alpha_result_t;
typedef struct exponent_scale14_t {ap_uint<1> sign;ap_int<7> weight; } exponent_scale14_t;
typedef ap_fixed<4,3> bias14_t;
typedef ap_ufixed<4,0,AP_RND_CONV,AP_SAT,0> layer9_t;
typedef ap_fixed<18,8> qrelu2_table_t;
typedef ap_fixed<15,10> qdense2_result_t;
typedef ap_fixed<4,3> weight10_t;
typedef ap_fixed<4,3> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,6> qdense2_alpha_result_t;
typedef struct exponent_scale15_t {ap_uint<1> sign;ap_int<3> weight; } exponent_scale15_t;
typedef ap_fixed<4,3> bias15_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_inv_table_t;


#endif
