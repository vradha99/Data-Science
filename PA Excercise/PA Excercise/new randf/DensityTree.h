#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <stdlib.h>
#include <math.h>

# define PI 3.14159265358979323846

using namespace cv;
using namespace std;

#ifndef DENSITYTREE_H
#define DENSITYTREE_H

class DensityTree
{
public:
    DensityTree();
    DensityTree(unsigned int D, unsigned int R, Mat X);
    void train();
    Mat densityXY();
	double* randomThreshold(int dimension, int idNode);
	double getInfGain(int idNode);
	void splitData(int dim, int idNode, double threshold);
private:
    unsigned int D;
    unsigned int n_thresholds;
	unsigned int numOfsamples;
	unsigned int dimensions;
	unsigned int nodesCount;
	unsigned int leafCount;
	unsigned int internalCount;
	double* det;
	unsigned int* sampleNumber;
	vector<int>* sampleIndex;
	Mat* nodes;
	double* thresholds;
	unsigned int* dimIndex;
	bool* isLeaf;
    Mat X;
       



};

#endif /* DENSITYTREE_H */
