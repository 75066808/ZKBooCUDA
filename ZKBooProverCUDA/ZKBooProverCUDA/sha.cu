#include "sha.cuh"
#include "utility.cuh"

__constant__ uint32_t dK[64] = { 
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
		0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
		0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
		0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6,
		0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3,
		0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138,
		0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e,
		0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
		0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
		0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
		0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814,
		0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 
};

__device__ void dShaInit(DSHA_CTX* ctx) {
	ctx->len = 0;
	ctx->bits = 0;

	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

__device__ void dShaTransform(DSHA_CTX* ctx) {
	uint32_t w[64];

	for (int j = 0; j < 16; j++) {
		w[j] = (ctx->data[j * 4] << 24) | (ctx->data[j * 4 + 1] << 16)
			| (ctx->data[j * 4 + 2] << 8) | ctx->data[j * 4 + 3];
	}

	uint32_t s0, s1;
	for (int j = 16; j < 64; j++) {
		s0 = RIGHTROTATE(w[j - 15], 7) ^ RIGHTROTATE(w[j - 15], 18)
			^ (w[j - 15] >> 3);
		s1 = RIGHTROTATE(w[j - 2], 17) ^ RIGHTROTATE(w[j - 2], 19)
			^ (w[j - 2] >> 10);
		w[j] = w[j - 16] + s0 + w[j - 7] + s1;
	}

	uint32_t a, b, c, d, e, f, g, h, temp1, temp2, maj;
	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

	for (int j = 0; j < 64; j++) {
		s1 = RIGHTROTATE(e, 6) ^ RIGHTROTATE(e, 11) ^ RIGHTROTATE(e, 25);

		temp1 = h + s1 + CH(e, f, g) + dK[j] + w[j];
		s0 = RIGHTROTATE(a, 2) ^ RIGHTROTATE(a, 13) ^ RIGHTROTATE(a, 22);

		maj = (a & (b ^ c)) ^ (b & c);
		temp2 = s0 + maj;

		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}

__device__ void dShaUpdate(DSHA_CTX* ctx, void* src, int size) {
	uint8_t* csrc = (uint8_t*)src;

	for (int i = 0; i < size; i++) {
		ctx->data[ctx->len] = csrc[i];
		ctx->len++;
		if (ctx->len == 64) {
			dShaTransform(ctx);
			ctx->len = 0;
			ctx->bits += 512;
		}
	}
}

__device__ void dShaFinal(DSHA_CTX* ctx, uint8_t* hash) {
	if (ctx->len < 56) {
		ctx->data[ctx->len] = 0x80;
		for (int i = ctx->len + 1; i < 56; i++)
			ctx->data[i] = 0x00;
	} else {
		ctx->data[ctx->len] = 0x80;
		for (int i = ctx->len + 1; i < 64; i++)
			ctx->data[i] = 0x00;

		dShaTransform(ctx);

		for (int i = 0; i < 56; i++)
			ctx->data[i] = 0x00;
	}

	ctx->bits += ctx->len * 8;

	ctx->data[63] = ctx->bits;
	ctx->data[62] = ctx->bits >> 8;
	ctx->data[61] = ctx->bits >> 16;
	ctx->data[60] = ctx->bits >> 24;
	ctx->data[59] = ctx->bits >> 32;
	ctx->data[58] = ctx->bits >> 40;
	ctx->data[57] = ctx->bits >> 48;
	ctx->data[56] = ctx->bits >> 56;

	dShaTransform(ctx);

	for (int i = 0; i < 8; i++) {
		hash[i * 4]     = (ctx->state[i] >> 24);
		hash[i * 4 + 1] = (ctx->state[i] >> 16);
		hash[i * 4 + 2] = (ctx->state[i] >> 8);
		hash[i * 4 + 3] = (ctx->state[i]);
	}
}