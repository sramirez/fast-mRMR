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

/** @file Rawdata.cpp
 *  @brief Used to handle the raw csv data.
 *
 *  Contains the RawData class and defines the basic
 *  datatypes for the project.
 *
 *  @author Iago Lastra (iagolast)
 */

#include "RawData.h"
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
/**
 * Constructor that creates a rawData object.
 *
 * @param data_table this is a matrix of bytes containing the translated csv data.
 * @param ds the number of data samples.
 * @param fs the number of features that each sample has.
 */
RawData::RawData() {
	dataFile = fopen("data.mrmr", "rb");
	calculateDSandFS();
	loadData();
	mallocGPU();
	moveGPU();
	calculateVR();
}

RawData::~RawData() {

}

/**
 * Free GPU data.
 */
void RawData::freeGPU() {
	cudaFree(d_data);
	cudaFree(d_acum);
}

/**
 *	Free CPU and GPU data.
 *
 *	Called only once at the end.
 */
void RawData::destroy() {
	freeGPU();
	free(valuesRange);
	free(h_data);
}

/**
 * Gets Datasize And Feature Size from mrmr File.
 */
void RawData::calculateDSandFS() {
	uint featuresSizeBuffer[1];
	uint datasizeBuffer[1];
	fread(datasizeBuffer, sizeof(uint), 1, dataFile);
	fread(featuresSizeBuffer, sizeof(uint), 1, dataFile);
	datasize = datasizeBuffer[0];
	featuresSize = featuresSizeBuffer[0];
	if( datasize % 16 != 0){
		printf("Error: dataset needs mod 16 patterns: %d patterns\n", datasize);
		exit(-1);
	}
}

/**
 *	Loads Data from mrmr File into memory.
 */
void RawData::loadData() {
	uint i, j;
	t_data buffer[1];
	h_data = (t_data*) calloc(featuresSize, sizeof(t_data) * datasize);
	fseek(dataFile, 8, 0);
	for (i = 0; i < datasize; i++) {
		for (j = 0; j < featuresSize; j++) {
			fread(buffer, sizeof(t_data), 1, dataFile);
			h_data[j * datasize + i] = buffer[0];
		}
	}
}

/**
 * Alloc space in the GPU to keep all data.
 *
 * End program on error.
 */
void RawData::mallocGPU() {
	cudaMalloc((void**) &d_acum, 255 * 255 * sizeof(uint));
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("Error allocating d_acum in GPU: %d", err);
		exit(-1);
	}
	cudaMalloc((void**) &d_data, datasize * featuresSize * sizeof(t_data));
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("Error allocating d_data in GPU: %d", err);
		exit(-1);
	}
}

/**
 * Moves the data from host to device.
 *
 * End program on error.
 */
void RawData::moveGPU() {
	cudaMemcpy(d_data, h_data, datasize * featuresSize * sizeof(t_data),
			cudaMemcpyHostToDevice);
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("Error moving data to GPU: %d", err);
		exit(-1);
	}
}

/**
 * Calculates how many different values has each feature.
 */
void RawData::calculateVR() {
	uint i, j;
	t_data dataReaded;
	uint vr;
	valuesRange = (uint*) calloc(featuresSize, sizeof(uint));
	for (i = 0; i < featuresSize; i++) {
		vr = 0;
		for (j = 0; j < datasize; j++) {
			dataReaded = h_data[i * datasize + j];
			if (dataReaded > vr) {
				vr++;
			}
		}
		valuesRange[i] = vr + 1;
	}
}

/**
 * @return gpu histogram to acumulate values in.
 *
 * It is kept in memory to avoid making a GPU malloc for each histogram.
 */
t_histogram RawData::getAcum() {
	return d_acum;
}

/**
 * @return The number of samples.
 */
uint RawData::getDataSize() {
	return datasize;
}

/**
 * @return The number of features.
 */
uint RawData::getFeaturesSize() {
	return featuresSize;
}

/**
 * Calculate the number of different values for a given feature.
 *
 * @param index : The feature index.
 * @return The number of values which a feature has FROM 1 to VALUES;
 */
uint RawData::getValuesRange(uint index) {
	return valuesRange[index];
}

/**
 *	@return An array containing the number of different values for each feature.
 */
uint * RawData::getValuesRangeArray() {
	return this->valuesRange;
}

/**
 * @return a vector containing a feature.
 */
t_feature RawData::getFeature(int index) {
	return h_data + index * datasize;
}

/**
 * @return the GPU vector that contains the feature.
 */
t_feature RawData::getFeatureGPU(int index) {
	return d_data + index * datasize;
}
