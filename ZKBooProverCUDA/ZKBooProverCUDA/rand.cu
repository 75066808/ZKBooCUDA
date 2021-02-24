#include "rand.cuh"

__device__ void dXorShift32Init(DXORSHIFT32_STATE* state, uint32_t seed) {
	state->a = seed;
}

__host__ void hXorShift32Init(DXORSHIFT32_STATE* state, uint32_t seed)
{
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

__host__ uint32_t hXorShift32Rand(DXORSHIFT32_STATE* state)
{
	uint32_t x = state->a;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	state->a = x;
	return x;
}


__device__ void getAllRandomness(unsigned char key[16], unsigned char randomness[RANDTAPE_SIZE], int offset) {
	unsigned char* counter = (unsigned char*)"0123456789012345";

	DAES_CTX ctx;
	dAesInit(&ctx, key, counter);

	unsigned char* plaintext = (unsigned char*)"0000000000000000";

	dAesEncryptCtr(&ctx, plaintext, &randomness[offset * 16], offset);
}

__host__ void hgetAllRandomness(unsigned char key[16], unsigned char randomness[RANDTAPE_SIZE], int offset)
{
	unsigned char* counter = (unsigned char*)"0123456789012345";

	DAES_CTX ctx;
	hAesInit(&ctx, key, counter);

	unsigned char* plaintext = (unsigned char*)"0000000000000000";

	hAesEncryptCtr(&ctx, plaintext, &randomness[offset * 16], offset);
}