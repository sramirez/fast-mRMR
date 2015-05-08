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


#include "JointProb.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "../cuda/histogramJoint.h"

JointProb::JointProb(RawData & rd, uint index1, uint index2) :
		rawData(rd) {
	this->index1 = index1;
	this->index2 = index2;
	this->valuesRange1 = rawData.getValuesRange(index1);
	this->valuesRange2 = rawData.getValuesRange(index2);
	this->datasize = rawData.getDataSize();
	this->h_acum = (t_histogram) calloc(valuesRange1 * valuesRange2,
			sizeof(uint));
	calculate();
}

JointProb::~JointProb() {
	free (h_acum);
}

/**
 * Calculate joint probability for two given features and save it in h_acum.
 */
void JointProb::calculate() {
	uint vr = valuesRange1 * valuesRange2;
	t_feature d_vector1 = rawData.getFeatureGPU(index1);
	t_feature d_vector2 = rawData.getFeatureGPU(index2);
	t_histogram d_acum = rawData.getAcum();
	histogramNaiveJoint(d_vector1, d_vector2, d_acum, datasize, 240,
			valuesRange2, vr);
	cudaMemcpy(h_acum, d_acum, vr * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("Error calculating Joint Prob in GPU: %d", err);
		exit(-1);
	}
}

/**
 * @return The probability that feature1 and feature2 have certain values.
 */
t_prob JointProb::getProb(t_data valueFeature1, t_data valueFeature2) {
	return (t_prob) h_acum[valueFeature1 * valuesRange2 + valueFeature2]
			/ (t_prob) datasize;
}

