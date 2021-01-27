#include "utility.cuh"
#include "mpc.cuh"

static const int GROUP_PER_WARP = WARP_SIZE / VERIFY_PATH_NUM;
static const int GROUP_PER_BLOCK = GROUP_PER_WARP * VERIFY_MPC_WARP_PER_BLOCK;

__constant__ uint32_t hA[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476,0xC3D2E1F0 };

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

__device__ void dMpcSha1Verify(int bgid, int gpid, unsigned char (*randomness)[RANDTAPE_SIZE], View* views, volatile bool* vflag) {
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

		if (*vflag)
			return;
	}

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

__device__ void output(View* v, void* result) {
	dMemcpy(result, &v->y[YSIZE - 5], 20);
}

__host__ void reconstruct(uint32_t* y0, uint32_t* y1, uint32_t* y2, uint32_t* result) {
	for (int i = 0; i < 8; i++) 
		result[i] = y0[i] ^ y1[i] ^ y2[i];
}