#include "utility.cuh"

__device__ void dMemcpy(void* dest, void* src, int bytes) {
	char* cdest = (char*)dest;
	char* csrc = (char*)src;

	for (int i = 0; i < bytes; i++)
		cdest[i] = csrc[i];
}

__device__ void dMemset(void* dest, int bytes, char ch) {
	char* cdest = (char*)dest;

	for (int i = 0; i < bytes; i++)
		cdest[i] = ch;
}

// __device__ void getAllRandomness(unsigned char key[16], unsigned char randomness[RANDTAPE_SIZE], int offset) {
// 	unsigned char* counter = (unsigned char*)"0123456789012345";

// 	DAES_CTX ctx;
// 	dAesInit(&ctx, key, counter);

// 	unsigned char* plaintext = (unsigned char*)"0000000000000000";

// 	dAesEncryptCtr(&ctx, plaintext, &randomness[offset * 16], offset);
// }

__device__ uint32_t getRandom32(unsigned char randomness[RANDTAPE_SIZE], int randCount) {
	uint32_t ret;
	dMemcpy(&ret, &randomness[randCount], 4);
	return ret;
}

__host__ uint32_t hgetRandom32(unsigned char randomness[RANDTAPE_SIZE], int randCount) {
	uint32_t ret;
	memcpy(&ret, &randomness[randCount], 4);
	return ret;
}


__device__ void H(unsigned char k[16], View* v, unsigned char r[4], uint8_t* hash) {
	DSHA_CTX ctx;
	dShaInit(&ctx);
	dShaUpdate(&ctx, k, 16);
	dShaUpdate(&ctx, v, sizeof(View));
	dShaUpdate(&ctx, r, 4);
	dShaFinal(&ctx, hash);
}

__host__ void hH(unsigned char k[16], View* v, unsigned char r[4], uint8_t* hash)
{
	DSHA_CTX ctx;
	hShaInit(&ctx);
	hShaUpdate(&ctx, k, 16);
	hShaUpdate(&ctx, v, sizeof(View));
	hShaUpdate(&ctx, r, 4);
	hShaFinal(&ctx, hash);
}

__host__ void H3(uint32_t y[8], a* as, int s, int* es) {
	uint8_t hash[32];

	DSHA_CTX ctx;
	hShaInit(&ctx);
	hShaUpdate(&ctx, y, 20);
	hShaUpdate(&ctx, as, sizeof(a) * s);
	hShaFinal(&ctx, hash);

	// Pick bits from hash
	int i = 0;
	int bitTracker = 0;
	while (i < s) {
		if (bitTracker >= 256) { // Generate new hash
			hShaInit(&ctx);
			hShaUpdate(&ctx, hash, sizeof(hash));
			hShaFinal(&ctx, hash);
			bitTracker = 0;
		}

		int b1 = GETBIT(hash[bitTracker / 8], bitTracker % 8);
		int b2 = GETBIT(hash[(bitTracker + 1) / 8], (bitTracker + 1) % 8);
		if (b1 == 0) {
			if (b2 == 0) {
				es[i] = 0;
				bitTracker += 2;
				i++;
			}
			else {
				es[i] = 1;
				bitTracker += 2;
				i++;
			}
		}
		else {
			if (b2 == 0) {
				es[i] = 2;
				bitTracker += 2;
				i++;
			}
			else {
				bitTracker += 2;
			}
		}
	}
}


