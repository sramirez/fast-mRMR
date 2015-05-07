/*
 * Histogram.cpp
 *
 *  Created on: Mar 20, 2014
 *      Author: iagolast
 */

#include "Histogram.h"
#include "../cuda/cuda_histogram.h"
#include <cuda_runtime_api.h>
#include <cuda.h>


Histogram::Histogram(RawData & rd) :
		rawData(rd) {
	h_acum = (t_histogram) calloc(255 * 255, sizeof(uint));
}

Histogram::~Histogram() {
	free(h_acum);
}

/**
 * Calculates the feature histogram.
 *
 * If the feature has 64 different values or less, uses
 * histogram 64 to compute, otherwise uses histogramNaive.
 *
 * @param index : The index for the feature.
 * @return h_acum : The histogram with the bin count.
 */
t_histogram Histogram::getHistogram(uint index) {
	uint vr = rawData.getValuesRange(index);
	t_histogram d_acum = rawData.getAcum();
	t_feature d_vector = rawData.getFeatureGPU(index);

	if (vr < 64) {
		histogram64(d_acum, d_vector, rawData.getDataSize());
	} else {
		histogramNaive(d_vector, d_acum, rawData.getDataSize(), 240);
	}
	cudaMemcpy(h_acum, d_acum, vr * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("Error copying histogram from GPU to CPU: %d\n", err);
	}
	return h_acum;
}
