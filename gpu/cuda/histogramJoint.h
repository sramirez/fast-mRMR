/*
 * histogram.h
 *
 *  Created on: 11/02/2014
 *      Author: iago
 */

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

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

/**
 * Used to reduce the number of registers used on the GPU.
 */
typedef struct  histoparams{
	t_data * data_vector1;
	t_data * data_vector2;
	histogram histo;
	uint datasize;
	int valuesRangeY;
	uint lap;
	uint maxBins;
} histoparams;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////


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

extern "C" void histogramNaiveJoint(t_data * d_vector1, t_data * d_vector2, histogram d_hist, unsigned int datasize, int blocks, int valuesRangeY, uint maxBins);


#endif
