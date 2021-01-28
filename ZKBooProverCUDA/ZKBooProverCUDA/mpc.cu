#include "utility.cuh"
#include "mpc.cuh"

static const int GROUP_PER_WARP = WARP_SIZE / PATH_NUM;
static const int GROUP_PER_BLOCK = GROUP_PER_WARP * COMMIT_WARP_PER_BLOCK;

#ifdef SHA1
__constant__ uint32_t hA[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476,0xC3D2E1F0 };

#else
__constant__ uint32_t hA[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

__constant__ uint32_t k[64] = {
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
	0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#endif

__device__ void dMpcXor(int gpid, uint32_t *x, uint32_t *y, uint32_t *z) {
	z[gpid] = x[gpid] ^ y[gpid];
}

__device__ void dMpcAnd(int gpid, uint32_t *x, uint32_t *y, uint32_t *z, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, View* view, int* countY) {
	int gpid1 = (gpid + 1) % PATH_NUM;

	uint32_t r[2] = { getRandom32(randomness[gpid], *randCount), getRandom32(randomness[gpid1], *randCount) };
	*randCount += 4;

	z[gpid] = (x[gpid] & y[gpid1]) ^ (x[gpid1] & y[gpid]) ^ (x[gpid] & y[gpid]) ^ r[0] ^ r[1];

	view->y[*countY] = z[gpid];
	(*countY)++;
}

__device__ void dMpcNegate(int gpid, uint32_t *x, uint32_t *z) {
	z[gpid] = ~x[gpid];
}

__device__ void dMpcAdd(int bgid, int gpid, uint32_t *x, uint32_t *y, uint32_t *z, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, View* view, int* countY) {
	int gpid1 = (gpid + 1) % PATH_NUM;

	uint32_t c = 0;
	uint32_t r[2] = { getRandom32(randomness[gpid], *randCount), getRandom32(randomness[gpid1], *randCount) };
	*randCount += 4;

	__shared__ uint8_t as[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint8_t bs[GROUP_PER_BLOCK][PATH_NUM];

	uint8_t* a = as[bgid];
	uint8_t* b = bs[bgid];

	uint8_t t = 0;

	for (int i = 0; i < 31; i++) {
		a[gpid] = GETBIT(x[gpid] ^ c, i);
		b[gpid] = GETBIT(y[gpid] ^ c, i);

		t = (a[gpid] & b[gpid1]) ^ (a[gpid1] & b[gpid]) ^ (a[gpid] & b[gpid]) ^ GETBIT(r[1], i) ^ GETBIT(r[0], i) ^ t;
		
		c |= (t << (i + 1));
	}

	z[gpid] = x[gpid] ^ y[gpid] ^ c;

	view->y[*countY] = c;
	*countY += 1;
}

__device__ void dMpcAddK(int bgid, int gpid, uint32_t *x, uint32_t y, uint32_t *z, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, View* view, int* countY) {
	int gpid1 = (gpid + 1) % PATH_NUM;

	uint32_t c = 0;
	uint32_t r[2] = { getRandom32(randomness[gpid], *randCount), getRandom32(randomness[gpid1], *randCount) };
	*randCount += 4;

	__shared__ uint8_t as[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint8_t bs[GROUP_PER_BLOCK][PATH_NUM];

	uint8_t* a = as[bgid];
	uint8_t* b = bs[bgid];

	uint8_t t = 0;
	
	for (int i = 0; i < 31; i++) {
		a[gpid] = GETBIT(x[gpid] ^ c, i);
		b[gpid] = GETBIT(y ^ c, i);

		t = (a[gpid] & b[gpid1]) ^ (a[gpid1] & b[gpid]) ^ (a[gpid] & b[gpid]) ^ GETBIT(r[1], i) ^ GETBIT(r[0], i) ^ t;

		c |= (t << (i + 1));
	}


	z[gpid] = x[gpid] ^ y ^ c;

	view->y[*countY] = c;
	*countY += 1;
}

__device__ void dMpcRightRotate(int gpid, uint32_t *x, int i, uint32_t *z) {
	z[gpid] = RIGHTROTATE(x[gpid], i);
}

__device__ void dMpcLeftRotate(int gpid, uint32_t *x, int i, uint32_t *z) {
	z[gpid] = LEFTROTATE(x[gpid], i);
}

__device__ void dMpcRightShift(int gpid, uint32_t *x, int i, uint32_t *z) {
	z[gpid] = x[gpid] >> i;
}

__device__ void dMpcMAJ(int bgid, int gpid, uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *z, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, View* view, int* countY) {
	__shared__ uint32_t t0s[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t t1s[GROUP_PER_BLOCK][PATH_NUM];

	uint32_t* t0 = t0s[bgid];
	uint32_t* t1 = t1s[bgid];

	dMpcXor(gpid, a, b, t0);
	dMpcXor(gpid, a, c, t1);
	dMpcAnd(gpid, t0, t1, z, randomness, randCount, view, countY);
	dMpcXor(gpid, z, a, z);
}

__device__ void dMpcCH(int bgid, int gpid, uint32_t *e, uint32_t *f, uint32_t *g, uint32_t* z, unsigned char(*randomness)[RANDTAPE_SIZE], int* randCount, View* view, int* countY) {
	__shared__ uint32_t t0s[GROUP_PER_BLOCK][PATH_NUM];

	uint32_t* t0 = t0s[bgid];

	//e & (f^g) ^ g
	dMpcXor(gpid, f, g, t0);
	dMpcAnd(gpid, e, t0, t0, randomness, randCount, view, countY);
	dMpcXor(gpid, t0, g, z);
}

#ifdef SHA1
__device__ void dMpcSha(int bgid, int gpid, unsigned char* result, unsigned char* input, int numBits, unsigned char (*randomness)[RANDTAPE_SIZE], View* view, int* countY) {
	int randCount = 0;
	int chars = numBits >> 3;

	unsigned char chunk[64];

	__shared__ uint32_t ws[GROUP_PER_BLOCK][80][PATH_NUM];

	uint32_t(*w)[PATH_NUM] = ws[bgid];

	dMemcpy(chunk, input, chars);
	chunk[chars] = 0x80;

	dMemset(chunk + chars + 1, 61 - chars, 0);

	chunk[62] = numBits >> 8;
	chunk[63] = numBits;

	dMemcpy(view->x, chunk, 64);

	for (int j = 0; j < 16; j++) 
		w[j][gpid] = (chunk[j * 4] << 24)     |    
					 (chunk[j * 4 + 1] << 16) | 
					 (chunk[j * 4 + 2] << 8)  | 
					 (chunk[j * 4 + 3]);
	

	__shared__ uint32_t temps[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t t0s[GROUP_PER_BLOCK][PATH_NUM];

	uint32_t* temp = temps[bgid];
	uint32_t* t0 = t0s[bgid];

	for (int j = 16; j < 80; j++) {
		dMpcXor(gpid, w[j - 3], w[j - 8], temp);
		dMpcXor(gpid, temp, w[j - 14], temp);
		dMpcXor(gpid, temp, w[j - 16], temp);
		dMpcLeftRotate(gpid, temp, 1, w[j]);
	}

	__shared__ uint32_t as[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t bs[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t cs[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t ds[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t es[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t fs[GROUP_PER_BLOCK][PATH_NUM];

	uint32_t *a = as[bgid];
	uint32_t *b = bs[bgid];
	uint32_t *c = cs[bgid];
	uint32_t *d = ds[bgid];
	uint32_t *e = es[bgid];
	uint32_t *f = fs[bgid];

	uint32_t k;

	a[gpid] = hA[0];
	b[gpid] = hA[1];
	c[gpid] = hA[2];
	d[gpid] = hA[3];
	e[gpid] = hA[4];

	for (int i = 0; i < 80; i++) {
		if (i <= 19) {
			//f = d ^ (b & (c ^ d))
			dMpcXor(gpid, c, d, f);
			dMpcAnd(gpid, b, f, f, randomness, &randCount, view, countY);
			dMpcXor(gpid, d, f, f);
			k = 0x5A827999;
		}
		else if (i <= 39) {
			dMpcXor(gpid, b, c, f);
			dMpcXor(gpid, d, f, f);
			k = 0x6ED9EBA1;
		}
		else if (i <= 59) {
			//f = MAJ(b,c,d)
			dMpcMAJ(bgid, gpid, b, c, d, f, randomness, &randCount, view, countY);
			k = 0x8F1BBCDC;
		}
		else {
			dMpcXor(gpid, b, c, f);
			dMpcXor(gpid, d, f, f);
			k = 0xCA62C1D6;
		}

		//temp = (a leftrotate 5) + f + e + k + w[i]
		dMpcLeftRotate(gpid, a, 5, temp);
		dMpcAdd(bgid, gpid, f, temp, temp, randomness, &randCount, view, countY);
		dMpcAdd(bgid, gpid, e, temp, temp, randomness, &randCount, view, countY);
		dMpcAddK(bgid, gpid, temp, k, temp, randomness, &randCount, view, countY);
		dMpcAdd(bgid, gpid, w[i], temp, temp, randomness, &randCount, view, countY);

		e[gpid] = d[gpid];
		d[gpid] = c[gpid];
		dMpcLeftRotate(gpid, b, 30, c);
		b[gpid] = a[gpid];
		a[gpid] = temp[gpid];
	}

	__shared__ uint32_t hHas[GROUP_PER_BLOCK][5][PATH_NUM];

	uint32_t (*hHa)[PATH_NUM] = hHas[bgid];

	for (int i = 0; i < 5; i++)
		hHa[i][gpid] = hA[i];

	dMpcAdd(bgid, gpid, hHa[0], a, hHa[0], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[1], b, hHa[1], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[2], c, hHa[2], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[3], d, hHa[3], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[4], e, hHa[4], randomness, &randCount, view, countY);

	for (int i = 0; i < 5; i++) {
		dMpcRightShift(gpid, hHa[i], 24, t0);
		result[i * 4] = t0[gpid];

		dMpcRightShift(gpid, hHa[i], 16, t0);
		result[i * 4 + 1] = t0[gpid];

		dMpcRightShift(gpid, hHa[i], 8, t0);
		result[i * 4 + 2] = t0[gpid];

		result[i * 4 + 3] = hHa[i][gpid];
	}
}

#else 
__device__ void dMpcSha(int bgid, int gpid, unsigned char* result, unsigned char* input, int numBits, unsigned char(*randomness)[RANDTAPE_SIZE], View* view, int* countY) {
	int randCount = 0;
	int chars = numBits >> 3;

	unsigned char chunk[64];

	__shared__ uint32_t ws[GROUP_PER_BLOCK][64][PATH_NUM];

	uint32_t(*w)[PATH_NUM] = ws[bgid];

	dMemcpy(chunk, input, chars);
	chunk[chars] = 0x80;

	dMemset(chunk + chars + 1, 61 - chars, 0);

	chunk[62] = numBits >> 8;
	chunk[63] = numBits;

	dMemcpy(view->x, chunk, 64);

	for (int j = 0; j < 16; j++)
		w[j][gpid] = (chunk[j * 4] << 24) |
					 (chunk[j * 4 + 1] << 16) |
					 (chunk[j * 4 + 2] << 8) |
					 (chunk[j * 4 + 3]);


	__shared__ uint32_t s0s[GROUP_PER_BLOCK][PATH_NUM], s1s[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t t0s[GROUP_PER_BLOCK][PATH_NUM], t1s[GROUP_PER_BLOCK][PATH_NUM];

	uint32_t* s0 = s0s[bgid], * s1 = s1s[bgid];
	uint32_t* t0 = t0s[bgid], * t1 = t1s[bgid];

	for (int j = 16; j < 64; j++) {
		//s0[i] = RIGHTROTATE(w[i][j-15],7) ^ RIGHTROTATE(w[i][j-15],18) ^ (w[i][j-15] >> 3);
		dMpcRightRotate(gpid, w[j - 15], 7, t0);
		dMpcRightRotate(gpid, w[j - 15], 18, t1);
		dMpcXor(gpid, t0, t1, t0);
		dMpcRightShift(gpid, w[j - 15], 3, t1);
		dMpcXor(gpid, t0, t1, s0);

		//s1[i] = RIGHTROTATE(w[i][j-2],17) ^ RIGHTROTATE(w[i][j-2],19) ^ (w[i][j-2] >> 10);
		dMpcRightRotate(gpid, w[j - 2], 17, t0);
		dMpcRightRotate(gpid, w[j - 2], 19, t1);

		dMpcXor(gpid, t0, t1, t0);
		dMpcRightShift(gpid, w[j - 2], 10, t1);
		dMpcXor(gpid, t0, t1, s1);

		//w[i][j] = w[i][j-16]+s0[i]+w[i][j-7]+s1[i];
		dMpcAdd(bgid, gpid, w[j - 16], s0, t1, randomness, &randCount, view, countY);
		dMpcAdd(bgid, gpid, w[j - 7], t1, t1, randomness, &randCount, view, countY);
		dMpcAdd(bgid, gpid, t1, s1, w[j], randomness, &randCount, view, countY);
	}

	__shared__ uint32_t as[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t bs[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t cs[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t ds[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t es[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t fs[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t gs[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t hs[GROUP_PER_BLOCK][PATH_NUM];

	__shared__ uint32_t temp1s[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t temp2s[GROUP_PER_BLOCK][PATH_NUM];
	__shared__ uint32_t majs[GROUP_PER_BLOCK][PATH_NUM];

	uint32_t* a = as[bgid];
	uint32_t* b = bs[bgid];
	uint32_t* c = cs[bgid];
	uint32_t* d = ds[bgid];
	uint32_t* e = es[bgid];
	uint32_t* f = fs[bgid];
	uint32_t* g = gs[bgid];
	uint32_t* h = hs[bgid];

	uint32_t* temp1 = temp1s[bgid];
	uint32_t* temp2 = temp2s[bgid];
	uint32_t* maj = majs[bgid];

	a[gpid] = hA[0];
	b[gpid] = hA[1];
	c[gpid] = hA[2];
	d[gpid] = hA[3];
	e[gpid] = hA[4];
	f[gpid] = hA[5];
	g[gpid] = hA[6];
	h[gpid] = hA[7];

	for (int i = 0; i < 64; i++) {
		//s1 = RIGHTROTATE(e,6) ^ RIGHTROTATE(e,11) ^ RIGHTROTATE(e,25);
		dMpcRightRotate(gpid, e, 6, t0);
		dMpcRightRotate(gpid, e, 11, t1);
		dMpcXor(gpid, t0, t1, t0);

		dMpcRightRotate(gpid, e, 25, t1);
		dMpcXor(gpid, t0, t1, s1);

		//ch = (e & f) ^ ((~e) & g);
		//temp1 = h + s1 + CH(e,f,g) + k[i]+w[i];

		//t0 = h + s1

		dMpcAdd(bgid, gpid, h, s1, t0, randomness, &randCount, view, countY);
		dMpcCH(bgid, gpid, e, f, g, t1, randomness, &randCount, view, countY);

		//t1 = t0 + t1 (h+s1+ch)
		dMpcAdd(bgid, gpid, t0, t1, t1, randomness, &randCount, view, countY);
		dMpcAddK(bgid, gpid, t1, k[i], t1, randomness, &randCount, view, countY);
		dMpcAdd(bgid, gpid, t1, w[i], temp1, randomness, &randCount, view, countY);

		//s0 = RIGHTROTATE(a,2) ^ RIGHTROTATE(a,13) ^ RIGHTROTATE(a,22);
		dMpcRightRotate(gpid, a, 2, t0);
		dMpcRightRotate(gpid, a, 13, t1);
		dMpcXor(gpid, t0, t1, t0);
		dMpcRightRotate(gpid, a, 22, t1);
		dMpcXor(gpid, t0, t1, s0);

		dMpcMAJ(bgid, gpid, a, b, c, maj, randomness, &randCount, view, countY);

		//temp2 = s0+maj;
		dMpcAdd(bgid, gpid, s0, maj, temp2, randomness, &randCount, view, countY);

		h[gpid] = g[gpid];
		g[gpid] = f[gpid];
		f[gpid] = e[gpid];

		//e = d+temp1;
		dMpcAdd(bgid, gpid, d, temp1, e, randomness, &randCount, view, countY);

		d[gpid] = c[gpid];
		c[gpid] = b[gpid];
		b[gpid] = a[gpid];

		//a = temp1+temp2;
		dMpcAdd(bgid, gpid, temp1, temp2, a, randomness, &randCount, view, countY);
	}

	__shared__ uint32_t hHas[GROUP_PER_BLOCK][8][PATH_NUM];

	uint32_t(*hHa)[PATH_NUM] = hHas[bgid];

	for (int i = 0; i < 8; i++)
		hHa[i][gpid] = hA[i];

	dMpcAdd(bgid, gpid, hHa[0], a, hHa[0], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[1], b, hHa[1], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[2], c, hHa[2], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[3], d, hHa[3], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[4], e, hHa[4], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[5], f, hHa[5], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[6], g, hHa[6], randomness, &randCount, view, countY);
	dMpcAdd(bgid, gpid, hHa[7], h, hHa[7], randomness, &randCount, view, countY);

	for (int i = 0; i < 8; i++) {
		dMpcRightShift(gpid, hHa[i], 24, t0);
		result[i * 4] = t0[gpid];

		dMpcRightShift(gpid, hHa[i], 16, t0);
		result[i * 4 + 1] = t0[gpid];

		dMpcRightShift(gpid, hHa[i], 8, t0);
		result[i * 4 + 2] = t0[gpid];

		result[i * 4 + 3] = hHa[i][gpid];
	}
}

#endif

__device__ void output(View* v, void* result) {
	dMemcpy(result, &v->y[YSIZE - RESULT_WORD], RESULT_BYTE);
}

