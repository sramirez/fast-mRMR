/*
 * ProbTable.h
 *
 *  Created on: Mar 20, 2014
 *      Author: iagolast
 */

#ifndef PROBTABLE_H_
#define PROBTABLE_H_
#include "Histogram.h"

class ProbTable {
public:
	ProbTable(RawData & rawData);
	virtual ~ProbTable();
	void calculate();
	t_prob getProbability(uint feature, t_data value);
	void destroy();
private:
	t_prob_table table;
	uint* valuesRange;
	uint featuresSize;
	uint datasize;
	RawData rawData;
};

#endif /* PROBTABLE_H_ */
