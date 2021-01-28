#pragma once
//#define SHA1

#ifdef  SHA1
#define YSIZE						370
#define GENRAND_ITER				92
#define RESULT_WORD					5

#else
#define YSIZE						736
#define GENRAND_ITER				182
#define RESULT_WORD					8

#endif

#define WARP_SIZE					32

#define GENRAND_WARP_PER_BLOCK		4

#define VERIFY_HASH_WARP_PER_BLOCK  1

#define VERIFY_MPC_WARP_PER_BLOCK   1

#define NUM_ROUNDS					136

#define PATH_NUM					3

#define	VERIFY_PATH_NUM				2

#define RANDTAPE_SIZE				GENRAND_ITER * 16

#define RESULT_BYTE					RESULT_WORD * 4