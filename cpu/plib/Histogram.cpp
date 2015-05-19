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

#include "Histogram.h"

Histogram::Histogram(RawData & rd): rawData(rd) {

}

Histogram::~Histogram() {
}

//Calculates the histogram for the given feature index.
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
