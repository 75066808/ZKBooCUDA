#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "rand.cuh"
#include "utility.cuh"
#include "mpc.cuh"

__device__ unsigned char input[INPUT_MAX_SIZE];
__device__ unsigned char shares[NUM_ROUNDS][PATH_NUM][INPUT_MAX_SIZE];
__device__ unsigned char randomness[NUM_ROUNDS][PATH_NUM][RANDTAPE_SIZE];

__device__ unsigned char rs[NUM_ROUNDS][PATH_NUM][4];
__device__ unsigned char keys[NUM_ROUNDS][PATH_NUM][16];

__device__ View views[NUM_ROUNDS][PATH_NUM];

__device__ a as[NUM_ROUNDS];
__device__ z zs[NUM_ROUNDS];
__device__ int es[NUM_ROUNDS];

a ahs[NUM_ROUNDS];
z zhs[NUM_ROUNDS];
int ehs[NUM_ROUNDS];

//-------------------------------------------------------------------------

static const int GENKEY_THREAD_PER_BLOCK = WARP_SIZE * GENKEY_WARP_PER_BLOCK;
static const int GENKEY_BLOCK_PER_GRID = (NUM_ROUNDS * PATH_NUM - 1) / GENKEY_THREAD_PER_BLOCK + 1;

__global__ void generateKey(unsigned long long seed, int bytes) {
	int rtid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id in grid

	int rgid = rtid / PATH_NUM; // group id in grid
	int gpid = rtid % PATH_NUM; // path id in group

	if (rgid < NUM_ROUNDS) {
		DXORSHIFT32_STATE state;

		dXorShift32Init(&state, seed + rtid);

		for (int i = 0; i < 4; i++)
			rs[rgid][gpid][i] = dXorShift32Rand(&state);

		for (int i = 0; i < 16; i++)
			keys[rgid][gpid][i] = dXorShift32Rand(&state);

		for (int i = 0; i < bytes; i++)
			shares[rgid][gpid][i] = dXorShift32Rand(&state);
	}
}

//-------------------------------------------------------------------------

static const int GENRAND_THREAD_PER_BLOCK = WARP_SIZE * GENRAND_WARP_PER_BLOCK;
static const int GENRAND_BLOCK_PER_GRID = (NUM_ROUNDS * PATH_NUM * GENRAND_ITER - 1) / GENRAND_THREAD_PER_BLOCK + 1;

__global__ void generateRandom(void) {
	int rtid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id in grid

	int rgid = rtid / GENRAND_ITER / PATH_NUM; // group id in grid
	int gpid = rtid / GENRAND_ITER % PATH_NUM; // path id in group

	int piid = rtid % GENRAND_ITER;			   // iter id in path

	if (rgid < NUM_ROUNDS)
		getAllRandomness(keys[rgid][gpid], randomness[rgid][gpid], piid);

}

//-------------------------------------------------------------------------

static const int GROUP_PER_WARP = WARP_SIZE / PATH_NUM;
static const int GROUP_PER_BLOCK = GROUP_PER_WARP * COMMIT_WARP_PER_BLOCK;

static const int COMMIT_THREAD_PER_BLOCK = WARP_SIZE * COMMIT_WARP_PER_BLOCK;
static const int COMMIT_BLOCK_PER_GRID = (NUM_ROUNDS - 1) / GROUP_PER_BLOCK + 1;

__global__ void commitA(int bytes) {
	int rbid = blockIdx.x;	// block id in grid

	int bwid = threadIdx.x / WARP_SIZE;				 // warp  id in block
	int wgid = threadIdx.x % WARP_SIZE / PATH_NUM;   // group id in warp
	int gpid = threadIdx.x % WARP_SIZE % PATH_NUM;   // path  id in group

	int bgid = bwid * GROUP_PER_WARP + wgid;  // group id in block
	int rgid = rbid * GROUP_PER_BLOCK + bgid; // group id in grid

	if (wgid < GROUP_PER_WARP && rgid < NUM_ROUNDS) {

		if (gpid == 0) {
			for (int i = 0; i < bytes; i++)
				shares[rgid][2][i] = input[i] ^ shares[rgid][0][i] ^ shares[rgid][1][i];
		}

		unsigned char* input = shares[rgid][gpid];
		unsigned char hash[32];

		int countY = 0;

		dMpcSha(bgid, gpid, hash, input, bytes * 8, randomness[rgid], &views[rgid][gpid], &countY);

		for (int i = 0; i < RESULT_WORD; i++) {
			views[rgid][gpid].y[countY] = 
					(hash[i * 4] << 24)     | 
					(hash[i * 4 + 1] << 16) |
					(hash[i * 4 + 2] << 8)  | 
					(hash[i * 4 + 3]);

			countY += 1;
		}

		output(&views[rgid][gpid], as[rgid].yp[gpid]);

		unsigned char hash1[32];
		H(keys[rgid][gpid], &views[rgid][gpid], rs[rgid][gpid], hash1);

		dMemcpy(as[rgid].h[gpid], hash1, 32);
	}
}

//-------------------------------------------------------------------------

__host__ void generateE(void) {
	uint32_t finalHash[8];
	for (int i = 0; i < 8; i++)
		finalHash[i] = ahs[0].yp[0][i] ^ ahs[0].yp[1][i] ^ ahs[0].yp[2][i];

	H3(finalHash, ahs, NUM_ROUNDS, ehs);
}

//-------------------------------------------------------------------------

static const int PACK_THREAD_PER_BLOCK = WARP_SIZE * PACK_WARP_PER_BLOCK;
static const int PACK_BLOCK_PER_GRID = (NUM_ROUNDS * PATH_NUM - 1) / PACK_THREAD_PER_BLOCK + 1;

__global__ void packZ(void) {
	int rtid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id in grid

	int rgid = rtid / PATH_NUM; // group id in grid
	int gpid = rtid % PATH_NUM; // path id in group

	if (rgid < NUM_ROUNDS) {
		int index = (gpid - es[rgid] + 3) % 3;
		if (index != 2) {
			dMemcpy(zs[rgid].k[index], keys[rgid][gpid], 16);
			zs[rgid].v[index] = views[rgid][gpid];
			dMemcpy(zs[rgid].r[index], rs[rgid][gpid], 4);
		}
	}
}

//-------------------------------------------------------------------------

__host__ void writeToFile(void) {
	FILE* file;

	char outputFile[3 * sizeof(int) + 8];
	sprintf(outputFile, "out%i.bin", NUM_ROUNDS);
	file = fopen(outputFile, "wb");
	if (!file) {
		printf("Unable to open file!");
		return;
	}
	fwrite(ahs, sizeof(a), NUM_ROUNDS, file);
	fwrite(zhs, sizeof(z), NUM_ROUNDS, file);

	fclose(file);

}

//-------------------------------------------------------------------------

__global__ void warmup(void) {

}


// //-------------------------------------------------------------------------



int main() {
	warmup << < 1, 1 >> > ();

	char in[INPUT_MAX_SIZE];

	printf("Enter the string to be hashed (Max 55 characters): ");
	fgets(in, sizeof(in), stdin);

	int len = strlen(in) - 1;

	clock_t begin = clock();

	cudaMemcpyToSymbol(input, in, INPUT_MAX_SIZE);
	
	generateKey << <GENKEY_BLOCK_PER_GRID, GENKEY_THREAD_PER_BLOCK >> > (time(NULL), len);

	generateRandom << <GENRAND_BLOCK_PER_GRID, GENRAND_THREAD_PER_BLOCK >> > ();

	commitA << <COMMIT_BLOCK_PER_GRID, COMMIT_THREAD_PER_BLOCK >> > (len);

	cudaMemcpyFromSymbol(ahs, as, NUM_ROUNDS * sizeof(a));
	generateE(); // serial in cpu
	cudaMemcpyToSymbol(es, ehs, NUM_ROUNDS * sizeof(int));

	packZ << <PACK_BLOCK_PER_GRID, PACK_THREAD_PER_BLOCK >> > ();
	
	cudaMemcpyFromSymbol(zhs, zs, NUM_ROUNDS * sizeof(z));

	clock_t endGenerateProve = clock();

	writeToFile();

	clock_t end = clock();

	printf("Genereate Prove: %d ms\n", (endGenerateProve - begin) * 1000 / CLOCKS_PER_SEC);
	printf("Write To File: %d ms\n", (end - endGenerateProve) * 1000 / CLOCKS_PER_SEC);
	printf("Total: %d ms\n", (end - begin) * 1000 / CLOCKS_PER_SEC);

    return 0;
}
