#ifndef _RAND_CUH
#define _RAND_CHU

#include <stdint.h>
#include "aes.cuh"
#include "setting.cuh"

typedef struct  {
	uint32_t a;
} DXORSHIFT32_STATE;

__device__ void dXorShift32Init(DXORSHIFT32_STATE* state, uint32_t seed);

__device__ uint32_t dXorShift32Rand(DXORSHIFT32_STATE* state);

__device__ void getAllRandomness(unsigned char key[16], unsigned char randomness[RANDTAPE_SIZE], int offset);
// __device__ void getAllRandomness(unsigned char key[16], unsigned char randomness[1476], int offset);

#endif