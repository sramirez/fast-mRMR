/*
 * RawData.h
 *
 *  Created on: Mar 19, 2014
 *      Author: iagolast
 */

#ifndef RAWDATA_H_
#define RAWDATA_H_
#include "utils.h"
#include <fstream>
#include <iostream>
#include <vector>

class RawData {
public:
	RawData();
	virtual ~RawData();

	void calculateVR();
	void calculateDSandFS();
	void destroy();
	uint getValuesRange(uint index);
	uint* getValuesRangeArray();
	uint getDataSize();
	uint getFeaturesSize();
	t_feature getFeature(int index);

private:

	void loadData();
	t_data* data;
	uint featuresSize;
	uint datasize;
	uint* valuesRange;
	FILE * dataFile;

};

#endif /* RAWDATA_H_ */
