#pragma once

#include "device_launch_parameters.h"

#include <stdint.h>

typedef struct {
	uint32_t len;
	uint64_t bits;

	uint8_t data[64];
	uint32_t state[8];
} DSHA_CTX;

__device__ void dShaInit(DSHA_CTX* ctx);

__device__ void dShaUpdate(DSHA_CTX* ctx, void* src, int size);

__device__ void dShaFinal(DSHA_CTX* ctx, uint8_t* hash);