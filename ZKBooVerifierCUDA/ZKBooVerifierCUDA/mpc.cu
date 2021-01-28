#include "utility.cuh"
#include "mpc.cuh"

static const int GROUP_PER_WARP = WARP_SIZE / VERIFY_PATH_NUM;
static const int GROUP_PER_BLOCK = GROUP_PER_WARP * VERIFY_MPC_WARP_PER_BLOCK;

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

__device__ void dMpcAndVerify(int gpid, uint32_t *x, uint32_t *y, uint32_t *z, View* views, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, int* countY, volatile bool* vflag) {
	uint32_t r[2] = { getRandom32(randomness[0], *randCount), getRandom32(randomness[1], *randCount) };
	*randCount += 4;

	uint32_t vt = views[gpid].y[*countY];

	uint32_t t = (x[0] & y[1]) ^ (x[1] & y[0]) ^ (x[0] & y[0]) ^ r[0] ^ r[1];

	if (gpid == 0 && vt != t)
		*vflag = true;

	z[gpid] = vt;
	(*countY)++;
}

__device__ void dMpcNegate(int gpid, uint32_t *x, uint32_t *z) {
	z[gpid] = ~x[gpid];
}

__device__ void dMpcAddVerify(int bgid, int gpid, uint32_t *x, uint32_t *y, uint32_t *z, View* views, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, int* countY, volatile bool* vflag) {
	uint32_t r[2] = { getRandom32(randomness[0], *randCount), getRandom32(randomness[1], *randCount) };
	*randCount += 4;

	__shared__ uint8_t as[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint8_t bs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];

	uint8_t* a = as[bgid];
	uint8_t* b = bs[bgid];

	uint32_t vc = views[gpid].y[*countY];

	uint32_t c = 0;

	for (int i = 30; i >= 0; i--) {
		a[gpid] = GETBIT(x[gpid] ^ vc, i);
		b[gpid] = GETBIT(y[gpid] ^ vc, i);

		c |= (a[0] & b[1]) ^ (a[1] & b[0]) ^ (a[0] & b[0]) ^ GETBIT(r[1], i) ^ GETBIT(r[0], i) ^ GETBIT(vc, i);
		c <<= 1;
	}

	if (gpid == 0 && c != vc)
		*vflag = true;

	z[gpid] = x[gpid] ^ y[gpid] ^ vc;
	(*countY)++;
}

__device__ void dMpcAddKVerify(int bgid, int gpid, uint32_t* x, uint32_t y, uint32_t* z, View* views, unsigned char(*randomness)[RANDTAPE_SIZE], int* randCount, int* countY, volatile bool* vflag) {
	uint32_t r[2] = { getRandom32(randomness[0], *randCount), getRandom32(randomness[1], *randCount) };
	*randCount += 4;

	__shared__ uint8_t as[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint8_t bs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];

	uint8_t* a = as[bgid];
	uint8_t* b = bs[bgid];

	uint32_t vc = views[gpid].y[*countY];

	uint32_t c = 0;

	for (int i = 30; i >= 0; i--) {
		a[gpid] = GETBIT(x[gpid] ^ vc, i);
		b[gpid] = GETBIT(y ^ vc, i);

		c |= (a[0] & b[1]) ^ (a[1] & b[0]) ^ (a[0] & b[0]) ^ GETBIT(r[1], i) ^ GETBIT(r[0], i) ^ GETBIT(vc, i);
		c <<= 1;
	}

	if (gpid == 0 && c != vc)
		*vflag = true;

	z[gpid] = x[gpid] ^ y ^ vc;
	(*countY)++;
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

__device__ void dMpcMAJVerify(int bgid, int gpid, uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *z, View *views, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, int* countY, volatile bool* vflag) {
	__shared__ uint32_t t0s[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t t1s[GROUP_PER_BLOCK][VERIFY_PATH_NUM];

	uint32_t* t0 = t0s[bgid];
	uint32_t* t1 = t1s[bgid];

	dMpcXor(gpid, a, b, t0);
	dMpcXor(gpid, a, c, t1);
	dMpcAndVerify(gpid, t0, t1, z, views, randomness, randCount, countY, vflag);
	dMpcXor(gpid, z, a, z);
}

__device__ void dMpcCHVerify(int bgid, int gpid, uint32_t* e, uint32_t* f, uint32_t* g, uint32_t* z, View* views, unsigned char(*randomness)[RANDTAPE_SIZE], int* randCount, int* countY, volatile bool* vflag) {
	__shared__ uint32_t t0s[GROUP_PER_BLOCK][VERIFY_PATH_NUM];

	uint32_t* t0 = t0s[bgid];

	//e & (f^g) ^ g
	dMpcXor(gpid, f, g, t0);
	dMpcAndVerify(gpid, e, t0, t0, views, randomness, randCount, countY, vflag);
	dMpcXor(gpid, t0, g, z);
}


#ifdef SHA1 // sha1
__device__ void dMpcShaVerify(int bgid, int gpid, unsigned char (*randomness)[RANDTAPE_SIZE], View* views, volatile bool* vflag) {
	int randCount = 0;
	int countY = 0;

	__shared__ uint32_t ws[GROUP_PER_BLOCK][80][VERIFY_PATH_NUM];

	uint32_t(*w)[VERIFY_PATH_NUM] = ws[bgid];


	for (int j = 0; j < 16; j++)
		w[j][gpid] = (views[gpid].x[j * 4] << 24)     | 
					 (views[gpid].x[j * 4 + 1] << 16) | 
		             (views[gpid].x[j * 4 + 2] << 8)  | 
		             (views[gpid].x[j * 4 + 3]);


	__shared__ uint32_t temps[GROUP_PER_BLOCK][VERIFY_PATH_NUM];

	uint32_t* temp = temps[bgid];

	for (int j = 16; j < 80; j++) {
		dMpcXor(gpid, w[j - 3], w[j - 8], temp);
		dMpcXor(gpid, temp, w[j - 14], temp);
		dMpcXor(gpid, temp, w[j - 16], temp);
		dMpcLeftRotate(gpid, temp, 1, w[j]);
	}

	__shared__ uint32_t vas[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t vbs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t vcs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t vds[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t ves[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t fs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];

	uint32_t *va = vas[bgid];
	uint32_t *vb = vbs[bgid];
	uint32_t *vc = vcs[bgid];
	uint32_t *vd = vds[bgid];
	uint32_t *ve = ves[bgid];
	uint32_t *f = fs[bgid];

	uint32_t k;

	va[gpid] = hA[0];
	vb[gpid] = hA[1];
	vc[gpid] = hA[2];
	vd[gpid] = hA[3];
	ve[gpid] = hA[4];

	for (int i = 0; i < 80; i++) {
		if (i <= 19) {
			//f = d ^ (b & (c ^ d))
			dMpcXor(gpid, vc, vd, f);
			dMpcAndVerify(gpid, vb, f, f, views, randomness, &randCount, &countY, vflag);
			dMpcXor(gpid, vd, f, f);
			k = 0x5A827999;
		}
		else if (i <= 39) {
			dMpcXor(gpid, vb, vc, f);
			dMpcXor(gpid, vd, f, f);
			k = 0x6ED9EBA1;
		}
		else if (i <= 59) {
			//f = MAJ(b,c,d)
			dMpcMAJVerify(bgid, gpid, vb, vc, vd, f, views, randomness, &randCount, &countY, vflag);
			k = 0x8F1BBCDC;
		}
		else {
			dMpcXor(gpid, vb, vc, f);
			dMpcXor(gpid, vd, f, f);
			k = 0xCA62C1D6;
		}

		//temp = (a leftrotate 5) + f + e + k + w[i]
		dMpcLeftRotate(gpid, va, 5, temp);
		dMpcAddVerify(bgid, gpid, f, temp, temp, views, randomness, &randCount, &countY, vflag);
		dMpcAddVerify(bgid, gpid, ve, temp, temp, views, randomness, &randCount, &countY, vflag);
		dMpcAddKVerify(bgid, gpid, temp, k, temp, views, randomness, &randCount, &countY, vflag);
		dMpcAddVerify(bgid, gpid, w[i], temp, temp, views, randomness, &randCount, &countY, vflag);

		ve[gpid] = vd[gpid];
		vd[gpid] = vc[gpid];
		dMpcLeftRotate(gpid, vb, 30, vc);
		vb[gpid] = va[gpid];
		va[gpid] = temp[gpid];
	}


	if (*vflag)
		return;

	__shared__ uint32_t hHas[GROUP_PER_BLOCK][5][VERIFY_PATH_NUM];

	uint32_t (*hHa)[VERIFY_PATH_NUM] = hHas[bgid];

	for (int i = 0; i < 5; i++)
		hHa[i][gpid] = hA[i];

	dMpcAddVerify(bgid, gpid, hHa[0], va, hHa[0], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[1], vb, hHa[1], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[2], vc, hHa[2], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[3], vd, hHa[3], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[4], ve, hHa[4], views, randomness, &randCount, &countY, vflag);
}

#else  // sha256
__device__ void dMpcShaVerify(int bgid, int gpid, unsigned char(*randomness)[RANDTAPE_SIZE], View* views, volatile bool* vflag) {
	int randCount = 0;
	int countY = 0;

	__shared__ uint32_t ws[GROUP_PER_BLOCK][64][VERIFY_PATH_NUM];

	uint32_t(*w)[VERIFY_PATH_NUM] = ws[bgid];


	for (int j = 0; j < 16; j++)
		w[j][gpid] = (views[gpid].x[j * 4] << 24) |
					 (views[gpid].x[j * 4 + 1] << 16) |
					 (views[gpid].x[j * 4 + 2] << 8) |
					 (views[gpid].x[j * 4 + 3]);

	__shared__ uint32_t s0s[GROUP_PER_BLOCK][VERIFY_PATH_NUM], s1s[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t t0s[GROUP_PER_BLOCK][VERIFY_PATH_NUM], t1s[GROUP_PER_BLOCK][VERIFY_PATH_NUM];

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

		dMpcAddVerify(bgid, gpid, w[j - 16], s0, t1, views, randomness, &randCount, &countY, vflag);
		dMpcAddVerify(bgid, gpid, w[j - 7], t1, t1, views, randomness, &randCount, &countY, vflag);
		dMpcAddVerify(bgid, gpid, t1, s1, w[j], views, randomness, &randCount, &countY, vflag);
	}

	if (*vflag)
		return;

	__shared__ uint32_t vas[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t vbs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t vcs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t vds[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t ves[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t vfs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t vgs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t vhs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];

	__shared__ uint32_t temp1s[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t temp2s[GROUP_PER_BLOCK][VERIFY_PATH_NUM];
	__shared__ uint32_t majs[GROUP_PER_BLOCK][VERIFY_PATH_NUM];


	uint32_t* va = vas[bgid];
	uint32_t* vb = vbs[bgid];
	uint32_t* vc = vcs[bgid];
	uint32_t* vd = vds[bgid];
	uint32_t* ve = ves[bgid];
	uint32_t* vf = vfs[bgid];
	uint32_t* vg = vgs[bgid];
	uint32_t* vh = vhs[bgid];

	uint32_t* temp1 = temp1s[bgid];
	uint32_t* temp2 = temp2s[bgid];
	uint32_t* maj = majs[bgid];

	va[gpid] = hA[0];
	vb[gpid] = hA[1];
	vc[gpid] = hA[2];
	vd[gpid] = hA[3];
	ve[gpid] = hA[4];
	vf[gpid] = hA[5];
	vg[gpid] = hA[6];
	vh[gpid] = hA[7];

	for (int i = 0; i < 64; i++) {
		//s1 = RIGHTROTATE(e,6) ^ RIGHTROTATE(e,11) ^ RIGHTROTATE(e,25);
		dMpcRightRotate(gpid, ve, 6, t0);
		dMpcRightRotate(gpid, ve, 11, t1);
		dMpcXor(gpid, t0, t1, t0);

		dMpcRightRotate(gpid, ve, 25, t1);
		dMpcXor(gpid, t0, t1, s1);

		//ch = (e & f) ^ ((~e) & g);
		//temp1 = h + s1 + CH(e,f,g) + k[i]+w[i];

		//t0 = h + s1

		dMpcAddVerify(bgid, gpid, vh, s1, t0, views, randomness, &randCount, &countY, vflag);
		dMpcCHVerify(bgid, gpid, ve, vf, vg, t1, views, randomness, &randCount, &countY, vflag);

		//t1 = t0 + t1 (h+s1+ch)
		dMpcAddVerify(bgid, gpid, t0, t1, t1, views, randomness, &randCount, &countY, vflag);
		dMpcAddKVerify(bgid, gpid, t1, k[i], t1, views, randomness, &randCount, &countY, vflag);
		dMpcAddVerify(bgid, gpid, t1, w[i], temp1, views, randomness, &randCount, &countY, vflag);

		//s0 = RIGHTROTATE(a,2) ^ RIGHTROTATE(a,13) ^ RIGHTROTATE(a,22);
		dMpcRightRotate(gpid, va, 2, t0);
		dMpcRightRotate(gpid, va, 13, t1);
		dMpcXor(gpid, t0, t1, t0);
		dMpcRightRotate(gpid, va, 22, t1);
		dMpcXor(gpid, t0, t1, s0);

		dMpcMAJVerify(bgid, gpid, va, vb, vc, maj, views, randomness, &randCount, &countY, vflag);

		//temp2 = s0+maj;
		dMpcAddVerify(bgid, gpid, s0, maj, temp2, views, randomness, &randCount, &countY, vflag);

		vh[gpid] = vg[gpid];
		vg[gpid] = vf[gpid];
		vf[gpid] = ve[gpid];

		//e = d+temp1;
		dMpcAddVerify(bgid, gpid, vd, temp1, ve, views, randomness, &randCount, &countY, vflag);

		vd[gpid] = vc[gpid];
		vc[gpid] = vb[gpid];
		vb[gpid] = va[gpid];

		//a = temp1+temp2;
		dMpcAddVerify(bgid, gpid, temp1, temp2, va, views, randomness, &randCount, &countY, vflag);
	}

	if (*vflag)
		return;

	__shared__ uint32_t hHas[GROUP_PER_BLOCK][8][VERIFY_PATH_NUM];

	uint32_t(*hHa)[VERIFY_PATH_NUM] = hHas[bgid];

	for (int i = 0; i < 8; i++)
		hHa[i][gpid] = hA[i];

	dMpcAddVerify(bgid, gpid, hHa[0], va, hHa[0], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[1], vb, hHa[1], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[2], vc, hHa[2], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[3], vd, hHa[3], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[4], ve, hHa[4], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[5], vf, hHa[5], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[6], vg, hHa[6], views, randomness, &randCount, &countY, vflag);
	dMpcAddVerify(bgid, gpid, hHa[7], vh, hHa[7], views, randomness, &randCount, &countY, vflag);
}


#endif

__device__ void output(View* v, void* result) {
	dMemcpy(result, &v->y[YSIZE - RESULT_WORD], RESULT_BYTE);
}

__host__ void reconstruct(uint32_t* y0, uint32_t* y1, uint32_t* y2, uint32_t* result) {
	for (int i = 0; i < RESULT_WORD; i++)
		result[i] = y0[i] ^ y1[i] ^ y2[i];
}