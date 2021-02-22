#ifndef _AES_CUH
#define _AES_CUH

#include <stdint.h>

typedef struct{
	uint8_t key[16];
	uint8_t counter[16];
} DAES_CTX;

// // key size = 128 bits
// // plaintext size = 128 bits
// // ciphertext size = 128 bits

__device__ void dAesInit(DAES_CTX *ctx, uint8_t key[16], uint8_t counter[16]);

__device__ void dAesEncryptCtr(DAES_CTX* ctx, uint8_t plain[16], uint8_t cipher[16], uint32_t th);

#endif
