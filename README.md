This is an improved greedy implementation of the classical feature selection method: minimum Redundancy and Maximum Relevance (mRMR); presented by Peng in [1]. 

## Main features

Several optimizations have been introduced  in order to speed up the mutual information calculation, among them: 

- **Cache marginal probabilities**: Instead of obtaining the marginal probabilities in each calculation of mutual information, those are computed only once at the beginning of the program, and cached.

- **Accumulating redundancy**: This is the most important step of the optimization. Instead of calculating the mutual information between each candidate feature and every selected feature in S, now redundancy is accumulated in each iteration (greedy approach).

- **Data access pattern**: The access pattern of mRMR to the dataset is by feature, in contrast to many other ML (machine learning) algorithms, in which access pattern is row-wise. Although being a low-level technical nuance, this aspect can significantly degrade mRMR performance since random access has a much greater cost than block-wise access. This is specially important in the case of GPU, since data has to be transferred from CPU memory to GPU global memory. Here, we reorganize the way in which data is stored in memory, changing it to a columnar format.

## Implementations

Here, we include implementations for this new proposal in several platforms. 

1. **Sequential version**: we provide a sequential implementation in C++ for CPU processing to be executed in a single machine that includes all previous optimizations.
2. **CUDA implementation**: we also provide a GPU implementation using CUDA in order to speedup the previous version by taking advantage of GPU thread-parallelism. 
3. **Apache Spark**: a Apache Spark's implementation of the algorithm is also included for large-scale problems, which require a distributed processing in order to obtain efficient solutions.

## Project structure:

* _cpu_: C++ code for CPU.
* _gpu_: CUDA code for GPU implementation.
* _spark_: Scala code for distributed processing in Apache Spark platform.
* _utils_: this contains a data reader program that transforms data in CSV format to the format required by fast-mRMR algorithm (in binary and columnar-wise format). It also includes a data generator method in case we want to generate synthetic data specifying the structure of this data.
* _examples_: a folder with examples for all versions implemented.   

## License

Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements. See the NOTICE file distributed with this work for additional information regarding copyright ownership. The ASF licenses this file to You under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## References

[1] _"Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy,"_ Hanchuan Peng, Fuhui Long, and Chris Ding IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 27, No. 8, pp.1226-1238, 2005.

For further information, please take a look at our wiki:

https://github.com/sramirez/fast-mRMR/wiki
