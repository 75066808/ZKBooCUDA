#pragma once

#include "device_launch_parameters.h"

#include <stdint.h>

typedef struct {
	uint8_t key[16];
	uint8_t counter[16];
} DAES_CTX;


__device__ void dAesInit(DAES_CTX *ctx, uint8_t key[16], uint8_t counter[16]);

__device__ void dAesEncryptCtr(DAES_CTX* ctx, uint8_t plain[16], uint8_t cipher[16], uint32_t th);
