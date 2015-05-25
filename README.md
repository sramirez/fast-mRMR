Welcome to the fast-mRMR wiki!

This is an improved implementation of the classical feature selection method: minimum Redundancy and Maximum Relevance (mRMR); presented by Peng in [1]. 

## Main features

Several optimizations have been introduced in this improved version in order to speed up the costliest computation of the original algorithm: Mutual Information (MI) calculations. These optimizations are described in the followings: 

- **Cache marginal probabilities**: Instead of obtaining the marginal probabilities in each MI computation, those are calculated only once at the beginning of the program, and cached to be re-used in the next iterations.

- **Accumulating redundancy**: The most important optimization is the greedy nature of the algorithm. Instead of computing the mutual information between every pair of features, now redundancy is accumulated in each iteration and the computations are performed between the last selected feature in S and each feature in non-selected set of attributes. 

- **Data access pattern**: The access pattern of mRMR to the dataset is thought to be feature-wise, in contrast to many other ML (machine learning) algorithms, in which access pattern is row-wise. Although being a low-level technical nuance, this aspect can significantly degrade mRMR performance since random access has a much greater cost than block-wise access. This is specially important in the case of GPU, since data has to be transferred from CPU memory to GPU global memory. Here, we reorganize the way in which data is stored in memory, changing it to a columnar format.

## Implementations

Here, we include several implementations for different platforms, in order to ease the application of our proposal. These are: 

1. **Sequential version**: we provide a basic implementation in C++ for CPU processing. This is designed to be executed in a single machine. This version includes all aforementioned optimizations.
2. **CUDA implementation**: we also provide a GPU implementation (using CUDA) with the aim of speeding up the previous version through GPU's thread-parallelism. 
3. **Apache Spark**: a Apache Spark's implementation of the algorithm is also included for large-scale problems, which require a distributed processing in order to obtain efficient solutions.

Please, for further information refer to our wiki: https://github.com/sramirez/fast-mRMR/wiki

## Project structure

The code is organized as follows:

* _cpu_: C++ code for CPU ([+info](https://github.com/sramirez/fast-mRMR/wiki/CPU-version)).
* _gpu_: CUDA code for GPU implementation ([+info](https://github.com/sramirez/fast-mRMR/wiki/GPU-version)).
* _spark_: Scala code for distributed processing in Apache Spark platform ([+info](https://github.com/sramirez/fast-mRMR/wiki/Spark-version)).
* _utils_: this folder contains a data reader program that transforms data in CSV format to the format required by fast-mRMR algorithm (in binary and columnar-wise format) ([+info](https://github.com/sramirez/fast-mRMR/wiki/Data-Reader)). It also includes a data generator method in case we want to generate synthetic data specifying the structure of this data.
* _examples_: a folder with examples for all versions implemented.   

 

## License

Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements. See the NOTICE file distributed with this work for additional information regarding copyright ownership. The ASF licenses this file to You under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## References

[1] _"Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy,"_ Hanchuan Peng, Fuhui Long, and Chris Ding IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 27, No. 8, pp.1226-1238, 2005.

