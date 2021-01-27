#pragma once

#include "device_launch_parameters.h"

#include <stdint.h>

#include "setting.cuh"

typedef struct {
	unsigned char x[64];
	uint32_t y[YSIZE];
} View;

typedef struct {
	uint32_t yp[3][8];
	unsigned char h[3][32];
} a;

typedef struct {
	unsigned char k[2][16];
	View v[2];
	unsigned char r[2][4];
} z;

__device__ void dMpcSha1(int bgid, int gpid, unsigned char* result, unsigned char* input, int numBits, unsigned char (*randomness)[RANDTAPE_SIZE], View* view, int* countY);

__device__ void output(View* v, void* result);