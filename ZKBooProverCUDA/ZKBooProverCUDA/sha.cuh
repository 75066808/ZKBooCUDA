#ifndef _SHA_CUH
#define _SHA_CUH

#include <stdint.h>
#include "utility.cuh"

typedef struct{
	uint32_t len;
	uint64_t bits;

	uint8_t data[64];
	uint32_t state[8];
}DSHA_CTX;

__device__ void dShaInit(DSHA_CTX* ctx);

__host__ void hShaInit(DSHA_CTX* ctx);

__device__ void dShaUpdate(DSHA_CTX* ctx, void* src, int size);

__host__ void hShaUpdate(DSHA_CTX* ctx, void* src, int size);

__device__ void dShaFinal(DSHA_CTX* ctx, uint8_t* hash);

__host__ void hShaFinal(DSHA_CTX* ctx, uint8_t* hash);

#endif