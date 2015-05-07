/*
 * Histogram.h
 *
 *  Created on: Mar 20, 2014
 *      Author: iagolast
 */

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_
#include "RawData.h"

class Histogram {
public:
	Histogram(RawData & rd);
	virtual ~Histogram();
	t_histogram getHistogram(uint index);
private:
	RawData rawData;
	t_histogram h_acum;
};

#endif /* HISTOGRAM_H_ */
