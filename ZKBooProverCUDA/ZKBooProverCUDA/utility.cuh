#pragma once

#include "device_launch_parameters.h"

#include <stdint.h>

#include "mpc.cuh"

#define CH(e,f,g) ((e & f) ^ ((~e) & g))

#define RIGHTROTATE(x,n) (((x) >> (n)) | ((x) << (32-(n))))
#define LEFTROTATE(x,n) (((x) << (n)) | ((x) >> (32-(n))))

#define GETBIT(x, i) (((x) >> (i)) & 0x01)
#define SETBIT(x, i, b)   x= (b)&1 ? (x)|(1 << (i)) : (x)&(~(1 << (i)))

__device__ void dMemcpy(void* dest, void* src, int bytes);

__device__ void dMemset(void* dest, int bytes, char ch);

__device__ void getAllRandomness(unsigned char key[16], unsigned char randomness[RANDTAPE_SIZE], int offset);

__device__ uint32_t getRandom32(unsigned char randomness[RANDTAPE_SIZE], int randCount);

__device__ void H(unsigned char k[16], View* v, unsigned char r[4], uint8_t* hash);

__host__ void H3(uint32_t y[8], a* as, int s, int* es);