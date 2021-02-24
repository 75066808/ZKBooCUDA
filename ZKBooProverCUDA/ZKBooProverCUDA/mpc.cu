#include "mpc.cuh"

static const int GROUP_PER_WARP = WARP_SIZE / PATH_NUM;
static const int GROUP_PER_BLOCK = GROUP_PER_WARP * COMMIT_WARP_PER_BLOCK;

// static const int COMMIT_THREAD_PER_BLOCK = WARP_SIZE * COMMIT_WARP_PER_BLOCK;
// static const int GROUP_PER_BLOCK = COMMIT_THREAD_PER_BLOCK / PATH_NUM;

// static const int COMMIT_BLOCK_PER_GRID = (NUM_ROUNDS - 1) / GROUP_PER_BLOCK + 1;

__constant__ uint32_t hA[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476,0xC3D2E1F0 };

__device__ void dMpcXor(int gpid, uint32_t *x, uint32_t *y, uint32_t *z) {
	z[gpid] = x[gpid] ^ y[gpid];
}

__device__ void dMpcXorOneCore(uint32_t *x, uint32_t *y, uint32_t *z)
{
	z[0] = x[0] ^ y[0];
	z[1] = x[1] ^ y[1];
	z[2] = x[2] ^ y[2];
}

__device__ void dMpcAnd(int gpid, uint32_t *x, uint32_t *y, uint32_t *z, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, View* view, int* countY) {
	int gpid1 = (gpid + 1) % PATH_NUM;

	uint32_t r[2] = { getRandom32(randomness[gpid], *randCount), getRandom32(randomness[gpid1], *randCount) };
	*randCount += 4;

	z[gpid] = (x[gpid] & y[gpid1]) ^ (x[gpid1] & y[gpid]) ^ (x[gpid] & y[gpid]) ^ r[0] ^ r[1];

	view->y[*countY] = z[gpid];
	(*countY)++;
}

__device__ void dMpcAndOneCore(uint32_t *x, uint32_t *y, uint32_t *z, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, View* views, int* countY)
{
	uint32_t r[3] = { getRandom32(randomness[0], *randCount), getRandom32(randomness[1], *randCount), getRandom32(randomness[2], *randCount)};
	*randCount += 4;
	uint32_t t[3] = { 0 };

	t[0] = (x[0] & y[1]) ^ (x[1] & y[0]) ^ (x[0] & y[0]) ^ r[0] ^ r[1];
	t[1] = (x[1] & y[2]) ^ (x[2] & y[1]) ^ (x[1] & y[1]) ^ r[1] ^ r[2];
	t[2] = (x[2] & y[0]) ^ (x[0] & y[2]) ^ (x[2] & y[2]) ^ r[2] ^ r[0];
	z[0] = t[0];
	z[1] = t[1];
	z[2] = t[2];

	views[0].y[*countY] = z[0];
	views[1].y[*countY] = z[1];
	views[2].y[*countY] = z[2];

	(*countY)++;
}

__device__ void dMpcNegate(int gpid, uint32_t *x, uint32_t *z) {
	z[gpid] = ~x[gpid];
}

__device__ void dMpcNegateOneCore(uint32_t *x, uint32_t *z)
{
	z[0] = ~x[0];
	z[1] = ~x[1];
	z[2] = ~x[2];
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


__device__ void dMpcAddOneCore(uint32_t *x, uint32_t *y, uint32_t *z, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, View* views, int* countY)
{
	uint32_t c[3] = { 0 };
	uint32_t r[3] = { getRandom32(randomness[0], *randCount), getRandom32(randomness[1], *randCount), getRandom32(randomness[2], *randCount)};
	*randCount += 4;

	uint8_t a[3], b[3];
	uint8_t t;

	for(int i=0;i<31;i++)
	{
		a[0]=GETBIT(x[0]^c[0],i);
		a[1]=GETBIT(x[1]^c[1],i);
		a[2]=GETBIT(x[2]^c[2],i);

		b[0]=GETBIT(y[0]^c[0],i);
		b[1]=GETBIT(y[1]^c[1],i);
		b[2]=GETBIT(y[2]^c[2],i);

		t = (a[0]&b[1]) ^ (a[1]&b[0]) ^ GETBIT(r[1],i);
		SETBIT(c[0],i+1, t ^ (a[0]&b[0]) ^ GETBIT(c[0],i) ^ GETBIT(r[0],i));

		t = (a[1]&b[2]) ^ (a[2]&b[1]) ^ GETBIT(r[2],i);
		SETBIT(c[1],i+1, t ^ (a[1]&b[1]) ^ GETBIT(c[1],i) ^ GETBIT(r[1],i));

		t = (a[2]&b[0]) ^ (a[0]&b[2]) ^ GETBIT(r[0],i);
		SETBIT(c[2],i+1, t ^ (a[2]&b[2]) ^ GETBIT(c[2],i) ^ GETBIT(r[2],i));
	}

	z[0]=x[0]^y[0]^c[0];
	z[1]=x[1]^y[1]^c[1];
	z[2]=x[2]^y[2]^c[2];

	views[0].y[*countY] = c[0];
	views[1].y[*countY] = c[1];
	views[2].y[*countY] = c[2];
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


__device__ void dMpcAddKOneCore(uint32_t *x, uint32_t y, uint32_t *z, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, View* views, int* countY)
{
	uint32_t c[3] = { 0 };
	uint32_t r[3] = { getRandom32(randomness[0], *randCount), getRandom32(randomness[1], *randCount), getRandom32(randomness[2], *randCount)};
	*randCount += 4;

	uint8_t a[3], b[3];
	uint8_t t;

	for(int i=0;i<31;i++)
	{
		a[0]=GETBIT(x[0]^c[0],i);
		a[1]=GETBIT(x[1]^c[1],i);
		a[2]=GETBIT(x[2]^c[2],i);

		b[0]=GETBIT(y^c[0],i);
		b[1]=GETBIT(y^c[1],i);
		b[2]=GETBIT(y^c[2],i);

		t = (a[0]&b[1]) ^ (a[1]&b[0]) ^ GETBIT(r[1],i);
		SETBIT(c[0],i+1, t ^ (a[0]&b[0]) ^ GETBIT(c[0],i) ^ GETBIT(r[0],i));

		t = (a[1]&b[2]) ^ (a[2]&b[1]) ^ GETBIT(r[2],i);
		SETBIT(c[1],i+1, t ^ (a[1]&b[1]) ^ GETBIT(c[1],i) ^ GETBIT(r[1],i));

		t = (a[2]&b[0]) ^ (a[0]&b[2]) ^ GETBIT(r[0],i);
		SETBIT(c[2],i+1, t ^ (a[2]&b[2]) ^ GETBIT(c[2],i) ^ GETBIT(r[2],i));
	}

	z[0]=x[0]^y^c[0];
	z[1]=x[1]^y^c[1];
	z[2]=x[2]^y^c[2];


	views[0].y[*countY] = c[0];
	views[1].y[*countY] = c[1];
	views[2].y[*countY] = c[2];
	*countY += 1;
}


__device__ void dMpcRightRotate(int gpid, uint32_t *x, int i, uint32_t *z) {
	z[gpid] = RIGHTROTATE(x[gpid], i);
}

__device__ void dMpcRightRotateOneCore(uint32_t *x, int i, uint32_t *z)
{
	z[0] = RIGHTROTATE(x[0], i);
	z[1] = RIGHTROTATE(x[1], i);
	z[2] = RIGHTROTATE(x[2], i);
}

__device__ void dMpcLeftRotate(int gpid, uint32_t *x, int i, uint32_t *z) {
	z[gpid] = LEFTROTATE(x[gpid], i);
}



__device__ void dMpcLeftRotateOneCore(uint32_t *x, int i, uint32_t *z)
{
	z[0] = LEFTROTATE(x[0], i);
	z[1] = LEFTROTATE(x[1], i);
	z[2] = LEFTROTATE(x[2], i);
}


__device__ void dMpcRightShift(int gpid, uint32_t *x, int i, uint32_t *z) {
	z[gpid] = x[gpid] >> i;
}

__device__ void dMpcRightShiftOneCore(uint32_t *x, int i, uint32_t *z)
{
	z[0] = x[0] >> i;
	z[1] = x[1] >> i;
	z[2] = x[2] >> i;
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

__device__ void dMpcMAJOneCore(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *z, unsigned char (*randomness)[RANDTAPE_SIZE], int* randCount, View* views, int* countY)
{
	uint32_t t0[3];
	uint32_t t1[3];

	dMpcXorOneCore(a, b, t0);
	dMpcXorOneCore(a, c, t1);
	dMpcAndOneCore(t0, t1, z, randomness, randCount, views, countY);
	dMpcXorOneCore(z, a, z);
}

__device__ void dMpcSha1OneCore(unsigned char (*results)[32], unsigned char* input[3], int numBits, unsigned char (*randomness)[RANDTAPE_SIZE], View views[3], int* countY)
{
	int randCount = 0;
	int chars = numBits >> 3;
	unsigned char chunks[3][64];
	uint32_t w[80][3];

	for(int i=0; i<3 ; i++)
	{
		dMemcpy(chunks[i], input[i], chars);
		chunks[i][chars] = 0x80;

		dMemset(chunks[i] + chars + 1, 61 - chars, 0);
		chunks[i][62] = numBits >> 8;
		chunks[i][63] = numBits;

		dMemcpy(views[i].x, chunks[i], 64);

		for (int j = 0; j < 16; j++) 
		{
			w[j][i] = (chunks[i][j * 4] << 24) | (chunks[i][j * 4 + 1] << 16)
							| (chunks[i][j * 4 + 2] << 8) | chunks[i][j * 4 + 3];
		}
	}

	uint32_t temp[3];
	uint32_t t0[3];

	for (int j = 16; j < 80; j++) {
		dMpcXorOneCore(w[j-3], w[j-8], temp);
		dMpcXorOneCore(temp, w[j-14], temp);
		dMpcXorOneCore(temp, w[j-16], temp);
		dMpcLeftRotateOneCore(temp,1,w[j]);
	}

	uint32_t a[3] = { hA[0],hA[0],hA[0] };
	uint32_t b[3] = { hA[1],hA[1],hA[1] };
	uint32_t c[3] = { hA[2],hA[2],hA[2] };
	uint32_t d[3] = { hA[3],hA[3],hA[3] };
	uint32_t e[3] = { hA[4],hA[4],hA[4] };
	uint32_t f[3];
	uint32_t k;

	for (int i = 0; i < 80; i++) {
		if(i <= 19) {
			//f = d ^ (b & (c ^ d))
			dMpcXorOneCore(c,d,f);
			dMpcAndOneCore(b, f, f, randomness, &randCount, views, countY);
			dMpcXorOneCore(d,f,f);
			k = 0x5A827999;
		}
		else if(i <= 39) {
			dMpcXorOneCore(b,c,f);
			dMpcXorOneCore(d,f,f);
			k = 0x6ED9EBA1;
		}
		else if(i <= 59) {
			//f = MAJ(b,c,d)

			dMpcMAJOneCore(b,c,d,f,randomness, &randCount, views, countY);

			k = 0x8F1BBCDC;
		}
		else {
			dMpcXorOneCore(b,c,f);
			dMpcXorOneCore(d,f,f);
			k = 0xCA62C1D6;
		}

		//temp = (a leftrotate 5) + f + e + k + w[i]
		dMpcLeftRotateOneCore(a,5,temp);
		dMpcAddOneCore(f,temp,temp,randomness, &randCount, views, countY);
		dMpcAddOneCore(e,temp,temp,randomness, &randCount, views, countY);
		dMpcAddKOneCore(temp,k,temp,randomness, &randCount, views, countY);
		dMpcAddOneCore(w[i],temp,temp,randomness, &randCount, views, countY);

		dMemcpy(e, d, sizeof(uint32_t) * 3);
		dMemcpy(d, c, sizeof(uint32_t) * 3);
		dMpcLeftRotateOneCore(b,30,c);
		dMemcpy(b, a, sizeof(uint32_t) * 3);
		dMemcpy(a, temp, sizeof(uint32_t) * 3);
	}

	uint32_t hHa[5][3] = { { hA[0],hA[0],hA[0]  }, { hA[1],hA[1],hA[1] }, { hA[2],hA[2],hA[2] }, { hA[3],hA[3],hA[3] },
			{ hA[4],hA[4],hA[4] }};

	dMpcAddOneCore(hHa[0], a, hHa[0], randomness, &randCount, views, countY);
	dMpcAddOneCore(hHa[1], b, hHa[1], randomness, &randCount, views, countY);
	dMpcAddOneCore(hHa[2], c, hHa[2], randomness, &randCount, views, countY);
	dMpcAddOneCore(hHa[3], d, hHa[3], randomness, &randCount, views, countY);
	dMpcAddOneCore(hHa[4], e, hHa[4], randomness, &randCount, views, countY);

	for (int i = 0; i < 5; i++) {
		dMpcRightShiftOneCore(hHa[i], 24, t0);
		results[0][i * 4] = t0[0];
		results[1][i * 4] = t0[1];
		results[2][i * 4] = t0[2];
		dMpcRightShiftOneCore(hHa[i], 16, t0);
		results[0][i * 4 + 1] = t0[0];
		results[1][i * 4 + 1] = t0[1];
		results[2][i * 4 + 1] = t0[2];
		dMpcRightShiftOneCore(hHa[i], 8, t0);
		results[0][i * 4 + 2] = t0[0];
		results[1][i * 4 + 2] = t0[1];
		results[2][i * 4 + 2] = t0[2];

		results[0][i * 4 + 3] = hHa[i][0];
		results[1][i * 4 + 3] = hHa[i][1];
		results[2][i * 4 + 3] = hHa[i][2];
	}
}

__device__ void dMpcSha1(int bgid, int gpid, unsigned char* result, unsigned char* input, int numBits, unsigned char (*randomness)[RANDTAPE_SIZE], View* view, int* countY) {
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

__device__ void output(View* v, void* result) {
	dMemcpy(result, &v->y[YSIZE - 5], 20);
}








