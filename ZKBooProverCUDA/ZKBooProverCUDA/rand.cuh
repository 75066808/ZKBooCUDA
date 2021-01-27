#pragma once

#include "device_launch_parameters.h"
#include <stdint.h>

typedef struct  {
	uint32_t a;
} DXORSHIFT32_STATE;

__device__ void dXorShift32Init(DXORSHIFT32_STATE* state, uint32_t seed);

__device__ uint32_t dXorShift32Rand(DXORSHIFT32_STATE* state);