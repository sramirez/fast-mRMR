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

#ifndef UTILS_H_
#define UTILS_H_

#include <sys/time.h>
#include <cstddef>
#include <stdlib.h>

#define MAX_BINS 255

typedef unsigned char byte;
typedef unsigned int uint;
typedef byte t_data;
typedef t_data* t_feature ;
typedef double t_prob;
typedef t_prob** t_prob_table;
typedef uint* t_histogram;
typedef t_data** data_table;

////////////////////////////////////////////////////////////////////////////////
// Timer
////////////////////////////////////////////////////////////////////////////////
struct Timer {

	struct timeval begin, end;

	/// Start measuring time
	void start() {
		gettimeofday(&begin, NULL);
	}

	/// Time in milliseconds
	double stop() {
		gettimeofday(&end, NULL);
		double secs = (double) ((end.tv_sec + (end.tv_usec / 1000000.0))
				- (begin.tv_sec + (begin.tv_usec / 1000000.0)));
		return secs * 1000.0;
	}

};

////////////////////////////////////////////////////////////////////////////////
// Command Line Options
////////////////////////////////////////////////////////////////////////////////
typedef struct options {
	uint classIndex;
	uint selectedFeatures;
	char* file;
} options;
#endif /* UTILS_H_ */
