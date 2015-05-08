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

#include "histogramJoint.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define WARP_SIZE	32		// Threads per warp.
#define WARPS_PER_BLOCK 6	// 2.1 max 8 blocks per sm and max 48 warps per sm.
#define THREADS_PER_BLOCK 192 // warp size * warps/block
#define MAX_BINS_PER_BLOCK 2048	// ~6KB shared mem per block. (1 bin = 4 bytes)

inline __device__ void addByte(uint * s_histo, byte x, byte y,
		histoparams params) {
	uint pos = (x * params.valuesRangeY + y);
	if (pos >= (params.lap * MAX_BINS_PER_BLOCK)
			and pos < ((params.lap + 1) * MAX_BINS_PER_BLOCK)) {
		atomicAdd(&s_histo[pos % MAX_BINS_PER_BLOCK], 1);
	}
}

inline __device__ void addWord(uint * s_histo, uint fourValuesX,
		uint fourValuesY, histoparams params) {
	uint i = 0;
#pragma unroll 4
	for (i = 0; i < 32; i += 8) {
		addByte(s_histo, /* x */((t_data) (fourValuesX >> i)),
		/* y */((t_data) (fourValuesY >> i)), params);
	}
}

__global__ void naiveHistoKernelJoint(histoparams params) {
	uint i;
	__shared__ unsigned int sharedHistogram[MAX_BINS_PER_BLOCK];

	//Inicializar el acum a cero.
	if (params.lap == 0) {
		i = threadIdx.x + blockIdx.x * blockDim.x;
		while (i < params.maxBins) {
			params.histo[i] = 0;
			i += blockDim.x * gridDim.x;
		}
	}

	i = threadIdx.x;
	while (i < MAX_BINS_PER_BLOCK) {
		sharedHistogram[i] = 0;
		i += blockDim.x;
	}
	__syncthreads();

	i = threadIdx.x + blockIdx.x * blockDim.x;
	//int offset = blockDim.x * gridDim.x;
	while (i < params.datasize / 16) {
		uint4 fourValues1 = ((uint4 *) params.data_vector1)[i];
		uint4 fourValues2 = ((uint4 *) params.data_vector2)[i];
		addWord(sharedHistogram, fourValues1.x, fourValues2.x, params);
		addWord(sharedHistogram, fourValues1.y, fourValues2.y, params);
		addWord(sharedHistogram, fourValues1.z, fourValues2.z, params);
		addWord(sharedHistogram, fourValues1.w, fourValues2.w, params);
		i += blockDim.x * gridDim.x;  //offset
	}
	__syncthreads();

	i = threadIdx.x;
	while (i < MAX_BINS_PER_BLOCK) {
		atomicAdd(&(params.histo[params.lap * MAX_BINS_PER_BLOCK + i]),
				sharedHistogram[i]);
		i += blockDim.x;
	}
}

extern "C" void histogramNaiveJoint(t_data * d_vector1, t_data * d_vector2,
		histogram d_hist, unsigned int datasize, int blocks,
		int valuesRangeVector2, uint maxBins) {
	histoparams params;
	params.data_vector1 = d_vector1;
	params.data_vector2 = d_vector2;
	params.histo = d_hist;
	params.datasize = datasize;
	params.valuesRangeY = valuesRangeVector2;
	params.maxBins = maxBins;
	for (int i = 0; i <= (maxBins / MAX_BINS_PER_BLOCK); i++) {
		params.lap = i;
		naiveHistoKernelJoint<<<blocks, THREADS_PER_BLOCK>>>(params);
	}
}
