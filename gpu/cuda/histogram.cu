/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/**
 * @file: histogram.cu
 * @brief: Contains a program to calculate histograms on the GPU.
 * Each warp computes a local histogram in shared memory which are finally mixed in global memory and returned.
 *
 */
#include "cuda_histogram.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define HISTOGRAM_WARP_WARP_SIZE	32		// Threads per warp.
#define WARPS_PER_BLOCK 6	// 2.1 max 8 blocks per sm and max 48 warps per sm.
#define THREADS_PER_BLOCK 192 // Warp
#define MAX_BINS 255		// 1KB shared mem per warp. (1 bin = 4 bytes)

/**
 * Sums one to its corresponding bin.
 *
 * @param: data: Byte with the bin to sum.
 * @param: sharedHisto: Histogram to compute the new bin.
 */
inline __device__ void addByte(byte data, histogram sharedHisto) {
	atomicAdd(&sharedHisto[data], 1);
}

/**
 * Calculates four bins at the same time, this is done because is better
 * to read from global memory in groups of 16 bytes.
 *
 * @param: sharedHisto: The per-warp shared histogram.
 * @param: fourValuesX: contais 4 bytes (each one is a value to compute).
 */
inline __device__ void addWord(histogram sharedHisto, uint fourValuesX) {
#pragma unroll 4
	for (byte i = 0; i < 4; i++) {
		addByte((byte) (fourValuesX >> (i * 8)), sharedHisto);
	}
}

/**
 * This is the main kernel to compute histograms.
 * First set memory to zero then computes the partial histograms and finally
 * reduces into global memory.
 */
__global__ void naiveHistoKernel_warp(t_data * data_vector, histogram histo,
		const unsigned int datasize) {
	int i;
	__shared__ unsigned int sharedHistogram[MAX_BINS * WARPS_PER_BLOCK];
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int wid = threadIdx.x / HISTOGRAM_WARP_WARP_SIZE;

// Init memory.
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < MAX_BINS) {
		histo[i] = 0;
		i += blockDim.x * gridDim.x;
	}
	for (i = threadIdx.x; i < MAX_BINS * WARPS_PER_BLOCK; i +=
			THREADS_PER_BLOCK) {
		sharedHistogram[i] = 0;
	}
	__syncthreads();

// Compute local histograms.
	i = tid;
	int offset = blockDim.x * gridDim.x;
	while (i < datasize / 16) {
		uint4 fourValuesX = ((uint4 *) data_vector)[i]; //read 16 bytes from global.
		addWord(&sharedHistogram[MAX_BINS * wid], fourValuesX.x);
		addWord(&sharedHistogram[MAX_BINS * wid], fourValuesX.y);
		addWord(&sharedHistogram[MAX_BINS * wid], fourValuesX.z);
		addWord(&sharedHistogram[MAX_BINS * wid], fourValuesX.w);
		i += offset;
	}
	__syncthreads();

// Merge sharedHistograms into histo.
	for (i = threadIdx.x; i < MAX_BINS; i += THREADS_PER_BLOCK) {
		uint acum = 0;
		for (uint j = 0; j < WARPS_PER_BLOCK; j++) {
			acum += sharedHistogram[MAX_BINS * j + i];
		}
		atomicAdd(&histo[i], acum);
	}

}

/**
 * This method encapsulates all the process.
 * @param: d_vector: Contains the feature to be calculated.
 * @param: d_hist: The gpu memory space to save the histogram.
 * @param: datasize: How many patterns are in the feature.
 * @param: blocks: the number of blocks that will be launched on the gpu.
 */
extern "C" void histogramNaive(t_data * d_vector, histogram d_hist,
		unsigned int datasize, int blocks) {
	naiveHistoKernel_warp<<<blocks, THREADS_PER_BLOCK>>>(d_vector, d_hist,
			datasize);
}
