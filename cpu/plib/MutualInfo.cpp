/*
 * MutualInfo.cpp
 *
 *  Created on: Mar 20, 2014
 *      Author: iagolast
 */

#include "MutualInfo.h"
#include <math.h>

MutualInfo::MutualInfo(RawData & rd, ProbTable & pt) :
		probTable(pt), rawData(rd) {
}

MutualInfo::~MutualInfo() {

}

double MutualInfo::get(uint featureIndex1, uint featureIndex2) {
	uint range1 = rawData.getValuesRange(featureIndex1);
	uint range2 = rawData.getValuesRange(featureIndex2);
	uint i, j;
	t_prob mInfo = 0;
	t_prob jointProb = 0;
	t_prob marginalX = 0;
	t_prob marginalY = 0;
	t_prob division = 0;

	JointProb jointProbTable = JointProb(rawData, featureIndex1, featureIndex2);
	for (i = 0; i < range1; i++) {
		for (j = 0; j < range2; j++) {
			jointProb = jointProbTable.getProb(i, j);
			if (jointProb != 0) {
				marginalX = probTable.getProbability(featureIndex1, i);
				marginalY = probTable.getProbability(featureIndex2, j);
				division = jointProb / (marginalX * marginalY);
				mInfo += jointProb * log2(division);
			}
		}
	}
	return mInfo;
}
