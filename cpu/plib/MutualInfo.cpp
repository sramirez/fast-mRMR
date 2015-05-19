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

#include "MutualInfo.h"
#include <math.h>

MutualInfo::MutualInfo(RawData & rd, ProbTable & pt) :
		probTable(pt), rawData(rd) {
}

MutualInfo::~MutualInfo() {

}

//Calculates the mutual information between the given features.
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
