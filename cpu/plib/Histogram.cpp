/*
 * Histogram.cpp
 *
 *  Created on: Mar 20, 2014
 *      Author: iagolast
 */

#include "Histogram.h"

Histogram::Histogram(RawData & rd): rawData(rd) {

}

Histogram::~Histogram() {
}

t_histogram Histogram::getHistogram(uint index) {
	uint vr = rawData.getValuesRange(index);
	t_feature data = rawData.getFeature(index);
	t_histogram h_acum = (t_histogram) malloc(vr * sizeof(uint));

	// Initialize to zero
	for (uint i = 0; i < vr; i++) {
		h_acum[i] = 0;
	}
	//Calculate Histogram
	for (uint i = 0; i < rawData.getDataSize(); i++) {
		h_acum[data[i]]++;
	}
	return h_acum;
}
