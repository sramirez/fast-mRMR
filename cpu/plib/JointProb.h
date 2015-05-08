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
