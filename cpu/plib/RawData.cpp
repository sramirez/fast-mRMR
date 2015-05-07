/** @file Rawdata.cpp
 *  @brief Used to handle the raw csv data.
 *
 *  Contains the RawData class and defines the basic
 *  datatypes for the project.
 *
 *  @author Iago Lastar (iagolast)
 */
#include "RawData.h"
#include <string.h>
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
	calculateVR();
}

RawData::~RawData() {

}

/**
 *
 */
void RawData::destroy() {
	free(valuesRange);
	free(data);
}

void RawData::calculateDSandFS() {
	uint featuresSizeBuffer[1];
	uint datasizeBuffer[1];
	fread(datasizeBuffer, sizeof(uint), 1, dataFile);
	fread(featuresSizeBuffer, sizeof(uint), 1, dataFile);
	datasize = datasizeBuffer[0];
	featuresSize = featuresSizeBuffer[0];
}

void RawData::loadData() {
	uint i, j;
	t_data buffer[1];
	//	Reservo espacio para SIZE punteros
	data = (t_data*) calloc(featuresSize, sizeof(t_data) * datasize);
	fseek(dataFile, 8, 0);
	for (i = 0; i < datasize; i++) {
		for (j = 0; j < featuresSize; j++) {
			fread(buffer, sizeof(t_data), 1, dataFile);
			data[j * datasize + i] = buffer[0];
		}
	}
}
/**
 * Calculates how many different values each feature has.
 */
void RawData::calculateVR() {
	uint i, j;
	t_data dataReaded;
	uint vr;
	valuesRange = (uint*) calloc(featuresSize, sizeof(uint));
	for (i = 0; i < featuresSize; i++) {
		vr = 0;
		for (j = 0; j < datasize; j++) {
			dataReaded = data[i * datasize + j];
			if (dataReaded > vr) {
				vr++;
			}
		}
		valuesRange[i] = vr + 1;
	}
}

/**
 *
 */
uint RawData::getDataSize() {
	return datasize;
}

/**
 *
 */
uint RawData::getFeaturesSize() {
	return featuresSize;
}

/**
 * Returns how much values has a features FROM 1 to VALUES;
 */
uint RawData::getValuesRange(uint index) {
	return valuesRange[index];
}

/**
 *
 */
uint * RawData::getValuesRangeArray() {
	return this->valuesRange;
}

/**
 * Returns a vector containing a feature.
 */
t_feature RawData::getFeature(int index) {
	return data + index * datasize;
}
