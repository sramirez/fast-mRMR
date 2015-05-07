/*
 * JointProb.h
 *
 *  Created on: Mar 20, 2014
 *      Author: iagolast
 */

#ifndef JOINTPROB_H_
#define JOINTPROB_H_
#include "RawData.h"

class JointProb {
public:
	JointProb(RawData & rd, uint index1, uint index2);
	virtual ~JointProb();
	void calculate();
	t_prob getProb(t_data valueFeature1, t_data valueFeature2);

private:
	RawData rawData;
	t_histogram data;
	uint datasize;
	uint valuesRange1;
	uint valuesRange2;
	uint index1;
	uint index2;
};

#endif /* JOINTPROB_H_ */
