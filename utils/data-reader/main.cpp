/** @file main.cpp
 *  @brief This contains a method to transform csv files into mrmr binary files.
 *
 *  @author Iago Lastra (IagoLast)
 */
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include "StringTokenizer.h"
#include <map>

using namespace std;
typedef unsigned char byte;

/**
 * 	@brief Function to check if a key apears in a map.
 *	@param key the key that you are looking for
 *	@param mymap the map where you are going to check.
 *	@returns A boolean value depending if the key apears or not.
 */
bool contains(string key, map<string, byte> mymap) {
	map<string, byte>::iterator it = mymap.find(key);
	if (it == mymap.end()) {
		return false;
	}
	return true;
}

/**
 *@brief Translates a csv file into a binary file.
 *@brief Each different value will be mapped into an integer
 *@brief value from 0 to 255.
 */
int main(int argc, char* argv[]) {
	int datasize = 0;
	uint featuresSize = 0;
	uint featurePos = 0;
	uint i = 0;
	uint limitMod16;
	string line;
	string token;
	byte data;
	vector<byte> lastCategory; //Cuantas categorias tiene cada feature.
	vector<map<string, byte> > translationsVector; //Vector de mapas de traduccion
	map<string, byte> translations; //mapa de traduccion string->byte ara cada feature

//Parse console arguments.
	if (argc < 2) {
		printf("Usage: <inputfilename> [options]\n");
		exit(-1);
	}
	char* inputFilename = argv[1];
	ifstream inputFile(inputFilename);
	ofstream outputFile("data.mrmr", ios::out | ios::binary);

//Count lines and features.
	if (inputFile.is_open()) {
//Ignore first line (headers from csv)
		getline(inputFile, line);
		StringTokenizer strtk(line, ",");
		while (strtk.hasMoreTokens()) {
			token = strtk.nextToken();
			featuresSize++;
		}
		while (getline(inputFile, line)) {
			++datasize;
		}
		datasize++;

//Only %16 datasize can be computed on GPU.
		limitMod16 = (datasize % 16);
		datasize = datasize - limitMod16;

//write datasize and featuresize:
		outputFile.write(reinterpret_cast<char*>(&datasize), sizeof(datasize));
		outputFile.write(reinterpret_cast<char*>(&featuresSize),
				sizeof(featuresSize));

//Set pointer into beginning of the file.
		inputFile.clear();
		inputFile.seekg(0, inputFile.beg);

//Initialize translation map.
		for (i = 0; i <= featuresSize; ++i) {
			map<string, byte> map;
			translationsVector.push_back(map);
			lastCategory.push_back(0);
		}

//Ignore first line (headers from csv) again.
		getline(inputFile, line);
		uint lineCount = 0;
//Read and translate file to binary.
		while (getline(inputFile, line)) {
			if (lineCount % 100000 == 0) {
				printf("...%d / %d\n", lineCount, datasize);
			}
			if (lineCount == (datasize)) {
				printf("Readed Samples: %d\n", lineCount);
				if (limitMod16 != 0) {
					printf("Last %d samples ignored.\n", limitMod16);
				}
				break;
			}
			featurePos = 0;
			StringTokenizer strtk(line, ",");
			while (strtk.hasMoreTokens()) {
				token = strtk.nextToken();
				featurePos++;
				if (!contains(token, translationsVector[featurePos])) {
					translationsVector[featurePos][token] =
							lastCategory[featurePos];
					lastCategory[featurePos]++;
				}
				data = translationsVector[featurePos][token];
				//TODO: Use a buffer to write data.
				outputFile.write(reinterpret_cast<char*>(&data), sizeof(byte));
			}
			lineCount++;

		}
		outputFile.flush();
		outputFile.close();
		inputFile.close();
	} else {
		cout << "Error loading file.\n";
		exit(-1);
	}
	return 0;

}
