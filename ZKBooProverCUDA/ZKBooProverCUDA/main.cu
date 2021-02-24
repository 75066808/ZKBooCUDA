#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "rand.cuh"
#include "setting.cuh"
#include "utility.cuh"
#include "mpc.cuh"

__device__ unsigned char input[INPUT_MAX_SIZE];                // 什么叫device的变量？
__device__ unsigned char shares[NUM_ROUNDS][PATH_NUM][INPUT_MAX_SIZE];   // 这个就是x
__device__ unsigned char randomness[NUM_ROUNDS][PATH_NUM][RANDTAPE_SIZE];

__device__ unsigned char keys[NUM_ROUNDS][PATH_NUM][16];    // 为什么只要16 bytes
__device__ unsigned char rs[NUM_ROUNDS][PATH_NUM][4];       // 这个是什么，为什么只要4 bytes // 这个rs有什么用？现在还不知道  

__device__ View views[NUM_ROUNDS][PATH_NUM];

__device__ a as[NUM_ROUNDS];
__device__ z zs[NUM_ROUNDS];
__device__ int es[NUM_ROUNDS];

a ahs[NUM_ROUNDS];
z zhs[NUM_ROUNDS];
int ehs[NUM_ROUNDS];

unsigned char hkeys[NUM_ROUNDS][PATH_NUM][16];
unsigned char hrs[NUM_ROUNDS][PATH_NUM][4];
View hviews[NUM_ROUNDS][PATH_NUM];
a has[NUM_ROUNDS];

// // 
// // 

// //-------------------------------------------------------------------------


// // #define WARP_SIZE			   32
// // #define GENKEY_WARP_PER_BLOCK   1
// // #define PATH_NUM			       3     
// // #define NUM_ROUNDS			   136    
// // 

static const int GENKEY_THREAD_PER_BLOCK = WARP_SIZE * GENKEY_WARP_PER_BLOCK;            // 32 threads
static const int GENKEY_BLOCK_PER_GRID = (NUM_ROUNDS * PATH_NUM - 1) / GENKEY_THREAD_PER_BLOCK + 1;  // 

__global__ void generateKey(unsigned long long seed, int bytes) {
	int rtid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id in grid

	int rgid = rtid / PATH_NUM; // group id in grid
	int gpid = rtid % PATH_NUM; // path id in group

	if (rgid < NUM_ROUNDS) 
	{
		DXORSHIFT32_STATE state;

		dXorShift32Init(&state, seed + rtid);

		for (int i = 0; i < 4; i++)
			rs[rgid][gpid][i] = dXorShift32Rand(&state);

		for (int i = 0; i < 16; i++)
			keys[rgid][gpid][i] = dXorShift32Rand(&state);

		// for (int i = 0; i < bytes; i++)
		// 	shares[rgid][gpid][i] = dXorShift32Rand(&state);    
	}
}

__global__ void generateKeyOneCore(unsigned long long seed, int bytes)
{
	for(int i=0 ;i<NUM_ROUNDS; i++)
	{
		DXORSHIFT32_STATE state;
		dXorShift32Init(&state, seed + i);

		for (int j=0; j < 4; j++)
		{
			rs[i][0][j] = dXorShift32Rand(&state);
			rs[i][1][j] = dXorShift32Rand(&state);
			rs[i][2][j] = dXorShift32Rand(&state);
		}

		for (int j=0; j < 16; j++)
		{
			keys[i][0][j] = dXorShift32Rand(&state);
			keys[i][1][j] = dXorShift32Rand(&state);
			keys[i][2][j] = dXorShift32Rand(&state);
		}
	}
}

__global__ void generateShares(unsigned long long seed, int bytes)
{
	int rtid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id in grid

	int rgid = rtid / PATH_NUM; // group id in grid
	int gpid = rtid % PATH_NUM; // path id in group

	if (rgid < NUM_ROUNDS) 
	{
		DXORSHIFT32_STATE state;

		dXorShift32Init(&state, seed + rtid);
		if(gpid < 2)
		{
			for (int i = 0; i < bytes; i++)
				shares[rgid][gpid][i] = dXorShift32Rand(&state);  
		}
		__syncthreads();

		if(gpid == 2)
		{
			for (int i = 0; i < bytes; i++)
				shares[rgid][gpid][i] = input[i] ^ shares[rgid][0][i] ^ shares[rgid][1][i];
		}
	}
}

__global__ void generateSharesOneCore(unsigned long long seed, int bytes)
{
	for(int i=0 ;i<NUM_ROUNDS; i++)
	{
		DXORSHIFT32_STATE state;
		dXorShift32Init(&state, seed + i);

		for (int j=0; j < bytes; j++)
		{
			shares[i][0][j] = dXorShift32Rand(&state);
			shares[i][1][j] = dXorShift32Rand(&state);
			shares[i][2][j] = input[i] ^ shares[i][0][j] ^ shares[i][1][j];
		}
	}
}


// //-------------------------------------------------------------------------

// WARP_SIZE                32
// GENRAND_WARP_PER_BLOCK   4
// GENRAND_THREAD_PER_BLOCK 32 * 4
// NUM_ROUNDS               136
// PATH_NUM					3
// GENRAND_ITER             92


static const int GENRAND_THREAD_PER_BLOCK = WARP_SIZE * GENRAND_WARP_PER_BLOCK;   
static const int GENRAND_BLOCK_PER_GRID = (NUM_ROUNDS * PATH_NUM * GENRAND_ITER - 1) / GENRAND_THREAD_PER_BLOCK + 1;


__global__ void generateRandom(void) 
{
	int rtid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id in grid

	int rgid = rtid / GENRAND_ITER / PATH_NUM; // group id in grid
	int gpid = rtid / GENRAND_ITER % PATH_NUM; // path id in group

	int piid = rtid % GENRAND_ITER;			   // iter id in path

	if (rgid < NUM_ROUNDS)
		getAllRandomness(keys[rgid][gpid], randomness[rgid][gpid], piid);
}


__global__ void generateRandomOneCore(void)
{
	for(int i=0; i< NUM_ROUNDS; i++)
	{
		for(int j=0; j<PATH_NUM; j++)
		{
			for(int k=0; k<GENRAND_ITER; k++)
			{
				getAllRandomness(keys[i][j], randomness[i][j], k);
			}
		}
	}
}

// //-------------------------------------------------------------------------

// WARP_SIZE		32
// PATH_NUM 		3
// COMMIT_WARP_PER_BLOCK		1
// GROUP_PER_WARP    10 
// GROUP_PER_BLOCK   10

static const int GROUP_PER_WARP = WARP_SIZE / PATH_NUM;
static const int GROUP_PER_BLOCK = GROUP_PER_WARP * COMMIT_WARP_PER_BLOCK;

// COMMIT_THREAD_PER_BLOCK     32
// COMMIT_BLOCK_PER_GRID       14

static const int COMMIT_THREAD_PER_BLOCK = WARP_SIZE * COMMIT_WARP_PER_BLOCK;
// static const int GROUP_PER_BLOCK = COMMIT_THREAD_PER_BLOCK / PATH_NUM;
static const int COMMIT_BLOCK_PER_GRID = (NUM_ROUNDS - 1) / GROUP_PER_BLOCK + 1;

// __global__ void commitA(int bytes) 
// {
// 	int rbid = blockIdx.x;	// block id in grid

// 	// int bwid = threadIdx.x / WARP_SIZE;				 // warp  id in block
// 	// int wgid = threadIdx.x % WARP_SIZE / PATH_NUM;   // group id in warp

// 	int gpid = threadIdx.x % PATH_NUM;   // path  id in group
// 	int bgid = threadIdx.x / PATH_NUM;  // group id in block

// 	int rgid = rbid * GROUP_PER_BLOCK + bgid; // group id in grid

// 	if (bgid < GROUP_PER_BLOCK && rgid < NUM_ROUNDS) {
// 		unsigned char* input = shares[rgid][gpid];
// 		unsigned char hash[32];

// 		int countY = 0;

// 		dMpcSha1(bgid, gpid, hash, input, bytes * 8, randomness[rgid], &views[rgid][gpid], &countY);

// 		for (int i = 0; i < 5; i++) {
// 			views[rgid][gpid].y[countY] = (hash[i * 4] << 24) | (hash[i * 4 + 1] << 16)
// 											| (hash[i * 4 + 2] << 8) | hash[i * 4 + 3];
// 			countY += 1;
// 		}

// 		output(&views[rgid][gpid], as[rgid].yp[gpid]);

// 		// unsigned char hash1[32];
// 		// H(keys[rgid][gpid], &views[rgid][gpid], rs[rgid][gpid], hash1);
// 		// dMemcpy(as[rgid].h[gpid], hash1, 32);
// 	}
// }

__global__ void commitA(int bytes) 
{
	int rbid = blockIdx.x;	// block id in grid

	int bwid = threadIdx.x / WARP_SIZE;				 // warp  id in block
	int wgid = threadIdx.x % WARP_SIZE / PATH_NUM;   // group id in warp
	int gpid = threadIdx.x % WARP_SIZE % PATH_NUM;   // path  id in group

	int bgid = bwid * GROUP_PER_WARP + wgid;  // group id in block
	int rgid = rbid * GROUP_PER_BLOCK + bgid; // group id in grid


	if (wgid < GROUP_PER_WARP && rgid < NUM_ROUNDS) {
		unsigned char* input = shares[rgid][gpid];
		unsigned char hash[32];

		int countY = 0;

		dMpcSha1(bgid, gpid, hash, input, bytes * 8, randomness[rgid], &views[rgid][gpid], &countY);

		for (int i = 0; i < 5; i++) {
			views[rgid][gpid].y[countY] = (hash[i * 4] << 24) | (hash[i * 4 + 1] << 16)
											| (hash[i * 4 + 2] << 8) | hash[i * 4 + 3];
			countY += 1;
		}

		output(&views[rgid][gpid], as[rgid].yp[gpid]);

		// unsigned char hash1[32];
		// H(keys[rgid][gpid], &views[rgid][gpid], rs[rgid][gpid], hash1);

		// dMemcpy(as[rgid].h[gpid], hash1, 32);
	}
}


__global__ void commitAOneCore(int bytes)
{
	for(int i=0; i < NUM_ROUNDS; i++)
	{
		unsigned char* inputs[3];
		inputs[0] = shares[i][0];
		inputs[1] = shares[i][1];
		inputs[2] = shares[i][2];

		unsigned char hashes[3][32];
		int countY = 0;
		dMpcSha1OneCore(hashes, inputs, bytes * 8, randomness[i], views[i], &countY);

		for (int j = 0; j < 5; j++) {
			views[i][0].y[countY] = (hashes[0][j * 4] << 24) | (hashes[0][j * 4 + 1] << 16)
											| (hashes[0][j * 4 + 2] << 8) | hashes[0][j * 4 + 3];

			views[i][1].y[countY] = (hashes[1][j * 4] << 24) | (hashes[1][j * 4 + 1] << 16)
											| (hashes[1][j * 4 + 2] << 8) | hashes[1][j * 4 + 3];

			views[i][2].y[countY] = (hashes[2][j * 4] << 24) | (hashes[2][j * 4 + 1] << 16)
											| (hashes[2][j * 4 + 2] << 8) | hashes[2][j * 4 + 3];
			countY += 1;
		}

		output(&views[i][0], as[i].yp[0]);
		output(&views[i][1], as[i].yp[1]);
		output(&views[i][2], as[i].yp[2]);
	}
}


__global__ void HashC()
{
	int rbid = blockIdx.x;	// block id in grid

	int bwid = threadIdx.x / WARP_SIZE;				 // warp  id in block
	int wgid = threadIdx.x % WARP_SIZE / PATH_NUM;   // group id in warp
	int gpid = threadIdx.x % WARP_SIZE % PATH_NUM;   // path  id in group

	int bgid = bwid * GROUP_PER_WARP + wgid;  // group id in block
	int rgid = rbid * GROUP_PER_BLOCK + bgid; // group id in grid

	if (wgid < GROUP_PER_WARP && rgid < NUM_ROUNDS) {

		unsigned char hash1[32];
		H(keys[rgid][gpid], &views[rgid][gpid], rs[rgid][gpid], hash1);

		dMemcpy(as[rgid].h[gpid], hash1, 32);
	}
}


__global__ void HashCOneCore()
{
	for(int i=0; i < NUM_ROUNDS; i++)
	{
		unsigned char hash1[32];
		H(keys[i][0], &views[i][0], rs[i][0], hash1);
		dMemcpy(as[i].h[0], hash1, 32);

		H(keys[i][1], &views[i][1], rs[i][1], hash1);
		dMemcpy(as[i].h[1], hash1, 32);
		
		H(keys[i][2], &views[i][2], rs[i][2], hash1);
		dMemcpy(as[i].h[2], hash1, 32);
	}
}

__host__ void hHashC()
{
	for(int i=0; i< NUM_ROUNDS; i++)
	{
		unsigned char hash1[32];
		hH(hkeys[i][0], &hviews[i][0], hrs[i][0], hash1);
		memcpy(has[i].h[0], hash1, 32);

		hH(hkeys[i][1], &hviews[i][1], hrs[i][1], hash1);
		memcpy(has[i].h[1], hash1, 32);
		
		hH(hkeys[i][2], &hviews[i][2], hrs[i][2], hash1);
		memcpy(has[i].h[2], hash1, 32);
	}
}


// //-------------------------------------------------------------------------

__host__ void generateE(void) {
	uint32_t finalHash[8];
	for (int i = 0; i < 8; i++)
		finalHash[i] = ahs[0].yp[0][i] ^ ahs[0].yp[1][i] ^ ahs[0].yp[2][i];

	H3(finalHash, ahs, NUM_ROUNDS, ehs);
}

// //-------------------------------------------------------------------------

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

__global__ void packZOneCore(void)
{
	for(int i=0; i<NUM_ROUNDS; i++)
	{
		int e = es[i];
		zs[i].v[0] = views[i][e];
		zs[i].v[1] = views[i][(e+1) % 3];
		dMemcpy(zs[i].k[0], keys[i][e], 16);
		dMemcpy(zs[i].k[1], keys[i][(e+1) % 3], 16);
		dMemcpy(zs[i].r[0], rs[i][e], 4);
		dMemcpy(zs[i].r[1], rs[i][(e+1) % 3], 4);
	}
}

// //-------------------------------------------------------------------------

// __host__ void writeToFile(void) {
// 	FILE* file;

// 	char outputFile[3 * sizeof(int) + 8];
// 	sprintf(outputFile, "out%i.bin", NUM_ROUNDS);
// 	file = fopen(outputFile, "wb");
// 	if (!file) {
// 		printf("Unable to open file!");
// 		return;
// 	}
// 	fwrite(ahs, sizeof(a), NUM_ROUNDS, file);
// 	fwrite(zhs, sizeof(z), NUM_ROUNDS, file);

// 	fclose(file);

// }

//-------------------------------------------------------------------------

__global__ void warmup(void) {
}


// //-------------------------------------------------------------------------



int main() {
	warmup << < 1, 1 >> > ();
	// printf("Hello World!");

	char in[INPUT_MAX_SIZE] = "12345678901234567890123456789012345678901234567890";

	// printf("Enter the string to be hashed (Max 55 characters): ");
	// fgets(in, sizeof(in), stdin);       

	int len = strlen(in) - 1;   
	
	// cuda events

	cudaEvent_t eventStart, eventEnd;
	cudaEvent_t eventKeysStart, eventKeysEnd;
	cudaEvent_t	eventSharesStart, eventSharesEnd;
	cudaEvent_t	eventRandomStart, eventRandomEnd;
	cudaEvent_t eventCommitStart, eventCommitEnd;
	cudaEvent_t eventHashStart, eventHashEnd;
	cudaEvent_t eventGenerateEStart, eventGenerateEEnd;
	cudaEvent_t eventPackZStart, eventPackZEnd;

	cudaEventCreate( &eventStart);
	cudaEventCreate( &eventEnd);

	cudaEventCreate( &eventKeysStart);
	cudaEventCreate( &eventKeysEnd);

    cudaEventCreate( &eventSharesStart);
	cudaEventCreate( &eventSharesEnd);

	cudaEventCreate( &eventRandomStart);
	cudaEventCreate( &eventRandomEnd);

	cudaEventCreate( &eventCommitStart);
	cudaEventCreate( &eventCommitEnd);

	cudaEventCreate( &eventHashStart);
	cudaEventCreate( &eventHashEnd);

	cudaEventCreate( &eventGenerateEStart);
	cudaEventCreate( &eventGenerateEEnd);

	cudaEventCreate( &eventPackZStart);
	cudaEventCreate( &eventPackZEnd);


	cudaEventRecord( eventStart, 0);
	cudaEventSynchronize(eventStart);
	// clock_t begin = clock();

	// 
	// 
	cudaMemcpyToSymbol(input, in, INPUT_MAX_SIZE);
	// cudaMemcpyToSymbol(input, in, INPUT_MAX_SIZE);    

	// 

	// Generate keys
	cudaEventRecord( eventKeysStart, 0);
	cudaEventSynchronize(eventKeysStart);

	// generateKey << <GENKEY_BLOCK_PER_GRID, GENKEY_THREAD_PER_BLOCK >> > (time(NULL), len);
	generateKeyOneCore<< <1, 1>> > (time(NULL), len);

	cudaEventRecord( eventKeysEnd, 0);
	cudaEventSynchronize(eventKeysEnd);
	float   TimeGenerateKeys;
    cudaEventElapsedTime( &TimeGenerateKeys, eventKeysStart, eventKeysEnd );
    // printf( "Time to generate keys:  %3.5f ms\n", TimeGenerateKeys );

	// Sharing secrets
	cudaEventRecord( eventSharesStart, 0 );
	cudaEventSynchronize(eventSharesStart);

	// generateShares<< <GENKEY_BLOCK_PER_GRID, GENKEY_THREAD_PER_BLOCK >> > (time(NULL), len);
	generateShares<< <1, 1>> >(time(NULL), len);

	cudaEventRecord( eventSharesEnd, 0);
	cudaEventSynchronize(eventSharesEnd);

	float   TimeGenerateShares;
    cudaEventElapsedTime( &TimeGenerateShares, eventSharesStart, eventSharesEnd );
    // printf( "Time to generate shares:  %3.5f ms\n", TimeGenerateShares );

	// generate random
	cudaEventRecord( eventRandomStart, 0 );
	cudaEventSynchronize(eventRandomStart);
	// generateRandom << <GENRAND_BLOCK_PER_GRID, GENRAND_THREAD_PER_BLOCK >> > ();
	generateRandomOneCore << <1, 1>> > ();

	cudaEventRecord( eventRandomEnd, 0);
	cudaEventSynchronize(eventRandomEnd);
	float   TimeGenerateRandom;
    cudaEventElapsedTime( &TimeGenerateRandom, eventRandomStart, eventRandomEnd );
    // printf( "Time to generate random:  %3.5f ms\n", TimeGenerateRandom );

	// 假设前面生成的sharing random  以及 key 都是正确的。
	// commit
	cudaEventRecord( eventCommitStart, 0);
	cudaEventSynchronize(eventCommitStart);
	// printf( "%d %d %d\n", COMMIT_BLOCK_PER_GRID, GROUP_PER_BLOCK, COMMIT_THREAD_PER_BLOCK);
	// commitA << <COMMIT_BLOCK_PER_GRID, COMMIT_THREAD_PER_BLOCK >> > (len);
	commitAOneCore<< <1, 1 >> > (len);

	cudaEventRecord( eventCommitEnd, 0);
	cudaEventSynchronize(eventCommitEnd);
	float   TimeGenerateCommit;
    cudaEventElapsedTime( &TimeGenerateCommit, eventCommitStart, eventCommitEnd );
    // printf( "Time to generate commit:  %3.5f ms\n", TimeGenerateCommit );


	// Hash

	// cudaMemcpyFromSymbol(hkeys, keys, NUM_ROUNDS * PATH_NUM * 16 * sizeof(unsigned char));
	// cudaMemcpyFromSymbol(hrs, rs, NUM_ROUNDS * PATH_NUM * 4 * sizeof(unsigned char));
	// cudaMemcpyFromSymbol(hviews, views, NUM_ROUNDS * PATH_NUM * sizeof(View));
	// cudaMemcpyFromSymbol(has, as, NUM_ROUNDS * sizeof(a));

	cudaEventRecord( eventHashStart, 0);
	cudaEventSynchronize(eventHashStart);
	// HashC << <COMMIT_BLOCK_PER_GRID, COMMIT_THREAD_PER_BLOCK >> > ();
	HashCOneCore << <1, 1>> >();
	// hHashC();

	cudaEventRecord( eventHashEnd, 0);
	cudaEventSynchronize(eventHashEnd);
	float   TimeGenerateHash;
    cudaEventElapsedTime( &TimeGenerateHash, eventHashStart, eventHashEnd );
    // printf( "Time to generate Hash:  %3.5f ms\n", TimeGenerateHash );

	cudaEventRecord( eventGenerateEStart, 0);
	cudaEventSynchronize(eventGenerateEStart);
	cudaMemcpyFromSymbol(ahs, as, NUM_ROUNDS * sizeof(a));
	generateE();                                             // serial in cpu
	cudaMemcpyToSymbol(es, ehs, NUM_ROUNDS * sizeof(int));
	cudaEventRecord( eventGenerateEEnd, 0);
	cudaEventSynchronize(eventGenerateEEnd);
	float   TimeGenerateE;
    cudaEventElapsedTime( &TimeGenerateE, eventGenerateEStart, eventGenerateEEnd );
    // printf( "Time to generate E:  %3.5f ms\n", TimeGenerateE );

	cudaEventRecord( eventPackZStart, 0);
	cudaEventSynchronize(eventPackZStart);
	// packZ << <PACK_BLOCK_PER_GRID, PACK_THREAD_PER_BLOCK >> > ();
	packZOneCore<< <1, 1>> >();
	
	cudaEventRecord( eventPackZEnd, 0);
	cudaEventSynchronize(eventPackZEnd);
	float   TimePackZ;
    cudaEventElapsedTime( &TimePackZ, eventPackZStart, eventPackZEnd );
    // printf( "Time to Pack Z:  %3.5f ms\n", TimePackZ );

	cudaMemcpyFromSymbol(zhs, zs, NUM_ROUNDS * sizeof(z));

	cudaEventRecord( eventEnd, 0);
	cudaEventSynchronize(eventEnd);
	float   TimeTotal;
    cudaEventElapsedTime( &TimeTotal, eventStart, eventEnd );

	printf( "Time to generate keys:  %3.5f ms\n", TimeGenerateKeys );
	printf( "Time to generate shares:  %3.5f ms\n", TimeGenerateShares );
	printf( "Time to generate random:  %3.5f ms\n", TimeGenerateRandom );
	printf( "Time to generate commit:  %3.5f ms\n", TimeGenerateCommit );

	printf( "Time to generate Hash:  %3.5f ms\n", TimeGenerateHash );
	printf( "Time to generate E:  %3.5f ms\n", TimeGenerateE );
	printf( "Time to Pack Z:  %3.5f ms\n", TimePackZ );
	printf( "Total: %3.5f ms\n", TimeTotal);


	// clock_t endGenerateProve = clock();

	// writeToFile();

	// clock_t end = clock();

	// printf("Genereate Prove: %d ms\n", (endGenerateProve - begin) * 1000 / CLOCKS_PER_SEC);
	// printf("Write To File: %d ms\n", (end - endGenerateProve) * 1000 / CLOCKS_PER_SEC);
	// printf("Total: %d ms\n", (end - begin) * 1000 / CLOCKS_PER_SEC);

    // return 0;
}
