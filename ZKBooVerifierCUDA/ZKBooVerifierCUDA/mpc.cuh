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

__device__ void dMpcSha1Verify(int bgid, int gpid, unsigned char(*randomness)[RANDTAPE_SIZE], View* views, volatile bool* vflag);

__device__ void output(View* v, void* result);

__host__ void reconstruct(uint32_t* y0, uint32_t* y1, uint32_t* y2, uint32_t* result);