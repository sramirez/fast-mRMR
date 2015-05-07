/*
 * utils.h
 *
 *  Created on: Mar 19, 2014
 *      Author: iagolast
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
} options;
#endif /* UTILS_H_ */
