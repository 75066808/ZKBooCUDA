#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "rand.cuh"
#include "utility.cuh"
#include "mpc.cuh"

__device__ unsigned char randomness[NUM_ROUNDS][VERIFY_PATH_NUM][RANDTAPE_SIZE];

__device__ a as[NUM_ROUNDS];
__device__ z zs[NUM_ROUNDS];
__device__ int es[NUM_ROUNDS];

a ahs[NUM_ROUNDS];
z zhs[NUM_ROUNDS];
int ehs[NUM_ROUNDS];

//-------------------------------------------------------------------------

__host__ void readFromFile(void) {
	FILE* file;
	char outputFile[3 * sizeof(int) + 8];

	sprintf(outputFile, "out%i.bin", NUM_ROUNDS);
	file = fopen(outputFile, "rb");

	if (!file) {
		printf("Unable to open file!");
		return;
	}

	fread(&ahs, sizeof(a), NUM_ROUNDS, file);
	fread(&zhs, sizeof(z), NUM_ROUNDS, file);
	fclose(file);
}

//-------------------------------------------------------------------------

__host__ void generateE(uint32_t y[8]) {
	H3(y, ahs, NUM_ROUNDS, ehs);
}

//-------------------------------------------------------------------------

static const int VERIFY_HASH_THREAD_PER_BLOCK = WARP_SIZE * VERIFY_HASH_WARP_PER_BLOCK;
static const int VERIFY_HASH_BLOCK_PER_GRID = (NUM_ROUNDS * VERIFY_PATH_NUM - 1) / VERIFY_HASH_THREAD_PER_BLOCK + 1;

__global__ void verifyHashAndResult(volatile bool* vflag) {
	if (*vflag)
		return;

	int rtid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id in grid
	int rgid = rtid / VERIFY_PATH_NUM; // group id in grid
	int gpid = rtid % VERIFY_PATH_NUM; // path  id in group

	if (rgid < NUM_ROUNDS) {
		uint8_t hash[32];
			
		z* z = &zs[rgid];
		a* a = &as[rgid];

		int index = (es[rgid] + gpid) % PATH_NUM;

		H(z->k[gpid], &z->v[gpid], z->r[gpid], hash);

		if (dMemcmp(a->h[index], hash, 32) != 0)
			*vflag = true;
		else {
			uint8_t result[20];
			output(&z->v[gpid], result);

			if (dMemcmp(a->yp[index], result, 20) != 0)
				*vflag = true;
		}
	}
	
}

//-------------------------------------------------------------------------

static const int GENRAND_THREAD_PER_BLOCK = WARP_SIZE * GENRAND_WARP_PER_BLOCK;
static const int GENRAND_BLOCK_PER_GRID = (NUM_ROUNDS * VERIFY_PATH_NUM * GENRAND_ITER - 1) / GENRAND_THREAD_PER_BLOCK + 1;

__global__ void generateRandom(void) {
	int rtid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id in grid
	int rgid = rtid / GENRAND_ITER / VERIFY_PATH_NUM;  // group id in grid
	int gpid = rtid / GENRAND_ITER % VERIFY_PATH_NUM;  // path  id in group

	int piid = rtid % GENRAND_ITER;					   // iter id in path

	if (rgid < NUM_ROUNDS)
		getAllRandomness(zs[rgid].k[gpid], randomness[rgid][gpid], piid);
}

//-------------------------------------------------------------------------

static const int VERIFY_MPC_THREAD_PER_BLOCK = WARP_SIZE * VERIFY_MPC_WARP_PER_BLOCK;
static const int VERIFY_MPC_BLOCK_PER_GRID = (NUM_ROUNDS * VERIFY_PATH_NUM * GENRAND_ITER - 1) / VERIFY_MPC_THREAD_PER_BLOCK + 1;

static const int GROUP_PER_WARP = WARP_SIZE / VERIFY_PATH_NUM;
static const int GROUP_PER_BLOCK = GROUP_PER_WARP * VERIFY_MPC_WARP_PER_BLOCK;

__global__ void verifyMPC(volatile bool* vflag) {
	if (*vflag)
		return;

	int rbid = blockIdx.x;	// block id in grid

	int bwid = threadIdx.x / WARP_SIZE;					    // warp  id in block
	int wgid = threadIdx.x % WARP_SIZE / VERIFY_PATH_NUM;   // group id in warp
	int gpid = threadIdx.x % WARP_SIZE % VERIFY_PATH_NUM;	// path  id in group

	int bgid = bwid * GROUP_PER_WARP + wgid;  // group id in block
	int rgid = rbid * GROUP_PER_BLOCK + bgid; // group id in grid

	if (rgid < NUM_ROUNDS)
		dMpcShaVerify(bgid, gpid, randomness[rgid], zs[rgid].v, vflag);
}

__global__ void warmup(void) {

}


//-------------------------------------------------------------------------


int main() {
	warmup << < 1, 1 >> > ();

	printf("Iterations of SHA: %d\n", NUM_ROUNDS);

	clock_t begin = clock();

	readFromFile();

	clock_t endReadFile = clock();

	uint32_t y[8];
	reconstruct(ahs[0].yp[0], ahs[0].yp[1], ahs[0].yp[2], y);

	generateE(y);

	clock_t endGenerateE = clock();

	volatile bool* vflag;

	bool vflagh = false;

	cudaMalloc(&vflag, sizeof(bool));
	cudaMemcpy((void*)vflag, &vflagh, sizeof(bool), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(as, ahs, NUM_ROUNDS * sizeof(a));
	cudaMemcpyToSymbol(zs, zhs, NUM_ROUNDS * sizeof(z));
	cudaMemcpyToSymbol(es, ehs, NUM_ROUNDS * sizeof(int));

	verifyHashAndResult<< <VERIFY_HASH_BLOCK_PER_GRID, VERIFY_HASH_THREAD_PER_BLOCK >> >(vflag);
	
	cudaMemcpy(&vflagh, (void*)vflag, sizeof(bool), cudaMemcpyDeviceToHost);

	if (!vflagh) {
		generateRandom << <GENRAND_BLOCK_PER_GRID, GENRAND_THREAD_PER_BLOCK >> > ();
		verifyMPC << <VERIFY_MPC_BLOCK_PER_GRID, VERIFY_MPC_THREAD_PER_BLOCK >> > (vflag);
		cudaMemcpy(&vflagh, (void*)vflag, sizeof(bool), cudaMemcpyDeviceToHost);
	}
	
	cudaFree((void*)vflag);

	clock_t end = clock();

	printf("Proof for hash: ");
	for (int i = 0; i < RESULT_WORD; i++)
		printf("%08X", y[i]);
	printf("\n");

	if (vflagh)
		printf("Not Verified\n");
	else
		printf("Verified\n");	
	
	printf("Read from file: %d ms\n", (endReadFile - begin) * 1000 / CLOCKS_PER_SEC);
	printf("Generate E: %d ms\n", (endGenerateE - endReadFile) * 1000 / CLOCKS_PER_SEC);
	printf("Verify : %d ms\n", (end - endGenerateE) * 1000 / CLOCKS_PER_SEC);
	printf("Total: %d ms\n", (end - begin) * 1000 / CLOCKS_PER_SEC);

    return 0;
}
