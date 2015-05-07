/*
 * MutualInfo.h
 *
 *  Created on: Mar 20, 2014
 *      Author: iagolast
 */

#ifndef MUTUALINFO_H_
#define MUTUALINFO_H_
#include "JointProb.h"
#include "ProbTable.h"

class MutualInfo {
public:
	MutualInfo(RawData & rd, ProbTable & pt);
	virtual ~MutualInfo();
	t_prob get(uint index1, uint f2);
private:
	RawData rawData;
	ProbTable probTable;
};

#endif /* MUTUALINFO_H_ */
