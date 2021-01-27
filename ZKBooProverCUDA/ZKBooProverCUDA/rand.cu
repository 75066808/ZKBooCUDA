#include "rand.cuh"

__device__ void dXorShift32Init(DXORSHIFT32_STATE* state, uint32_t seed) {
	state->a = seed;
}

__device__ uint32_t dXorShift32Rand(DXORSHIFT32_STATE* state) {
	uint32_t x = state->a;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	state->a = x;
	return x;
}
