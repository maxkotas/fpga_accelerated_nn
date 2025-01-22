//Numpy array shape [16]
//Min -0.500000000000
//Max 0.000000000000
//Number of zeros 6

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[16];
#else
bias2_t b2[16] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

#endif

#endif
