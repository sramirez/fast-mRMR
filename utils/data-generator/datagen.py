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


from random import randint  
BINS = 10
def create_doc(name, ds, fs, bins):
	f = open(name, 'w')
	for j in range(0, fs -1):
	    f.write('f' + str(j) + ',' )
	f.write('f' + str(ds) + '\n' )
	     
	for i in range(0, ds):
	    for j in range(0, fs -1):
		f.write(str(randint(0,bins))+ ',')
	    f.write(str(randint(0,bins))+ '\n')
	f.close()

"""create_doc('python_features_100', 80, 100, 10)
create_doc('python_features_1000', 80, 1000, 10)
create_doc('python_features_10000', 80, 10000, 10)
create_doc('python_features_100000', 80, 100000, 10)"""
create_doc('python_samples_400K_2K_15.csv', 400000, 2000, 15)
