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

#include "../plib/JointProb.h"
#include "../plib/MutualInfo.h"
#include "../plib/utils.h"

#include <vector>
#include <algorithm>
#include <limits>
#include <boost/bind.hpp>
#include <stdio.h>
#include <string.h>

using namespace std;

//Returns the index of the higher value in the classRelevances Vector different from classIndex
uint getMaxRelevance(vector<double> classRelevances, uint classIndex) {
	uint i = 0;
	uint newFeature = -1;
	double relevance = 0;
	for (i = 0; i < classRelevances.size(); ++i) {
		if (classRelevances[i] > relevance && i != classIndex) {
			relevance = classRelevances[i];
			newFeature = i;
		}
	}
	return newFeature;
}

options parseOptions(int argc, char*argv[]) {
	options opts;
	opts.classIndex = 0;
	opts.selectedFeatures = 10;
	opts.file = "../data.mrmr";

	if (argc > 1) {
		for (int i = 0; i < argc; ++i) {
			if (strcmp(argv[i], "-f") == 0) {
				opts.file = argv[i + 1];
			}
			if (strcmp(argv[i], "-a") == 0) {
				opts.selectedFeatures = atoi(argv[i + 1]) - 1;
			}
			if (strcmp(argv[i], "-c") == 0) {
				opts.classIndex = atoi(argv[i + 1]) - 1;
			}
			if (strcmp(argv[i], "-h") == 0) {
				printf(
						"fast-mrmr:\nOptions:\n -f <inputfile>\t\tMRMR file generated using mrmrReader (default: data.mrmr).\n-c <classindex>\t\tIndicates the class index in the dataset (default: 0).\n-a <nfeatures>\t Indicates the number of features to select (default: 10).\n-h Prints this message");
				exit(0);
			}
		}
	}
	return opts;
}

int main(int argc, char* argv[]) {
	options opts;
	uint i = 0;
	uint j = 0;
	uint newFeatureIndex = 0;
	uint lastFeatureIndex = 0;
	double mrmr = 0;
	double acum = 0;
	vector<double> relevances;
	vector<double> redundances;
	vector<int> selectedFeatures;

	Timer tm;
	opts = parseOptions(argc, argv);
	RawData rawData = RawData(opts.file);
	tm.start();
	ProbTable prob = ProbTable(rawData);
	MutualInfo mutualInfo = MutualInfo(rawData, prob);
	


	//Get relevances between all features and class.
	for (i = 0; i < rawData.getFeaturesSize(); ++i) {
		relevances.push_back(mutualInfo.get(opts.classIndex, i));
		redundances.push_back(0);
	}

	// Max relevance feature is added because no redundancy is possible.
	newFeatureIndex = getMaxRelevance(relevances, opts.classIndex);
	selectedFeatures.push_back(newFeatureIndex);
	lastFeatureIndex = newFeatureIndex;

	cout << newFeatureIndex << ",";
	//MRMR
	while (selectedFeatures.size() < rawData.getFeaturesSize() - 1 //-1 because class is discarded
	and selectedFeatures.size() < opts.selectedFeatures) {
		acum = -std::numeric_limits<double>::infinity();
		for (j = 0; j < rawData.getFeaturesSize(); ++j) {
			//If feature not in selected selectedFeatures
			if (find(selectedFeatures.begin(), selectedFeatures.end(), j)
					== selectedFeatures.end() && j != opts.classIndex) {
				redundances[j] += mutualInfo.get(lastFeatureIndex, j);
				mrmr = relevances[j]
						- (redundances[j] / selectedFeatures.size());
				if (mrmr > acum) {
					acum = mrmr;
					newFeatureIndex = j;
				}
			}
		}
		//Last feature doesn't prints comma.
		if ( (selectedFeatures.size() == (opts.selectedFeatures - 1)) or (selectedFeatures.size() == (rawData.getFeaturesSize() -2)) ){
			cout << newFeatureIndex;
		}else{
			cout << newFeatureIndex << ",";
		}
		selectedFeatures.push_back(newFeatureIndex);
		lastFeatureIndex = newFeatureIndex;
	}

	rawData.destroy();
	prob.destroy();
	printf("\n");
	return (0);
}
