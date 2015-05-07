/*
 * cuda_histogram.h
 *
 *  Created on: Apr 1, 2014
 *      Author: iagolast
 */

#ifndef CUDA_HISTOGRAM_H_
#define CUDA_HISTOGRAM_H_

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define HISTOGRAM64_BIN_COUNT 64
#define HISTOGRAM256_BIN_COUNT 256
#define UINT_BITS 32

typedef unsigned char byte;
typedef  uint *  histogram;
typedef unsigned int uint;
typedef unsigned char uchar;
typedef byte t_data;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

//May change on future hardware, so better parametrize the code
#define SHARED_MEMORY_BANKS 32

//Threadblock size: must be a multiple of (4 * SHARED_MEMORY_BANKS)
//because of the bit permutation of threadIdx.x
#define HISTOGRAM64_THREADBLOCK_SIZE  (4 * SHARED_MEMORY_BANKS)

//Warps ==subhistograms per threadblock
#define WARP_COUNT 6

//Threadblock size
#define HISTOGRAM256_THREADBLOCK_SIZE (WARP_COUNT * WARP_SIZE)

//Shared memory per threadblock
#define HISTOGRAM256_THREADBLOCK_MEMORY (WARP_COUNT * HISTOGRAM256_BIN_COUNT)

#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

////////////////////////////////////////////////////////////////////////////////
// GPU histogram
////////////////////////////////////////////////////////////////////////////////
extern "C" void histogram64(uint *d_Histogram, void *d_Data, uint byteCount);
extern "C" void histogramNaive(t_data * d_vector, histogram d_hist,
		unsigned int datasize, int blocks);
extern "C" void closeHistogram64(void);



#endif /* CUDA_HISTOGRAM_H_ */
