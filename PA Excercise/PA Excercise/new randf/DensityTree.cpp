
#include "DensityTree.h"
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <math.h>
#include <map>


using namespace cv;
using namespace std;
const double pi = 3.14159265358979323;
const double e = 2.71828182845904523;
const double inf = std::numeric_limits<double>::infinity();

DensityTree::DensityTree(unsigned int D, unsigned int n_thresholds, Mat X)
{
	this->D = D;
	this->X = X;
	this->n_thresholds = n_thresholds;

	numOfsamples = X.rows;
	dimensions = X.cols;

	nodesCount = pow(2, D) - 1;
	leafCount = pow(2, (D - 1));
	internalCount = nodesCount - leafCount;


	det = new double [nodesCount];
	sampleNumber = new unsigned int[nodesCount];
	sampleIndex = new vector<int> [nodesCount];

	vector<int> indizes;
	for (int i = 0; i < numOfsamples; i++)
	{
		indizes.push_back(i);
	}
	sampleIndex[0] = indizes;

	nodes = new Mat [nodesCount];

	nodes[0] = X.clone();
	sampleNumber[0] = numOfsamples;

	Mat Covar, Mu;
	calcCovarMatrix(nodes[0], Covar, Mu, CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_64F);

	det[0] = log(determinant(Covar));

	thresholds = new double [internalCount];
	dimIndex = new unsigned int [leafCount];


}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////




double* DensityTree::randomThreshold(int dimension, int idNode)
{
	double min;
	double max;

	minMaxLoc(nodes[idNode].col(dimension), &min, &max, NULL, NULL);
	double* randomThresholds = new double[n_thresholds];

	RNG rng(getTickCount());//Machine clock randomisation
	for (int i = 0; i < n_thresholds; i++)
	{
		randomThresholds[i] = rng.uniform(min, max);//uniform random thresholds b/w max and min

	}
	return randomThresholds;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DensityTree::splitData(int dim, int idNode, double threshold)
{
    int left=idNode *2 +1;
    int right=idNode*2 +2;

	Mat L ,R;

	vector<int> indL;
	vector<int> indR;

	int i = 0;
	while(i < sampleNumber[idNode])
	{
		if (nodes[idNode].at<double>(i, dim) <= threshold)
		{
			indL.push_back(sampleIndex[idNode][i]);
			L.push_back(nodes[idNode].row(i));
		}
		else
		{
			indR.push_back(sampleIndex[idNode][i]);
			R.push_back(nodes[idNode].row(i));
		}

		i++;
	}
	sampleIndex[left] = indL;
	sampleIndex[right] = indR;
	L.copyTo(nodes[left]);
	R.copyTo(nodes[right]);
	sampleNumber[left] = nodes[left].rows;
	sampleNumber[right] = nodes[right].rows;

     Mat CovL, ML;
	calcCovarMatrix(nodes[left], CovL, ML, CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_64F);
	det[left] = log(determinant(CovL));

	Mat CovR, MR;
	//cout << "S[2]" << S[2] << endl;
	calcCovarMatrix(nodes[right], CovR, MR, CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_64F);
	det[right] = log(determinant(CovR));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double DensityTree::getInfGain(int idNode)
{
    int left=idNode *2 +1;
    int right=idNode*2 +2;


double info= det[idNode] -(((double)sampleNumber[left] / (double)sampleNumber[idNode])*det[left]+((double)sampleNumber[right] / (double)sampleNumber[idNode])*det[right]);

//ig=log[|_/\_(sj)|]-SUM{sj/|sj|*log(_/\_(sj)|}#{L,R}

return info;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void DensityTree::train()
{

	for (int idNode = 0; idNode < internalCount; idNode++)
	{
		double max_OfinformationGain = -inf;
		double informationGain;
		double max_Threshold;
		int indexOfmaxDimension;
        int dim = 0;

		while(dim < dimensions)
		{
			double* threshold = randomThreshold(dim, idNode);//Calculate thresholds

			for (int id_th = 0; id_th < n_thresholds; id_th++)
			{
				splitData(dim, idNode, threshold[id_th]);//split the data
				informationGain = getInfGain(idNode);//find the info gain

				if (max_OfinformationGain < informationGain && informationGain != inf)
				{
					max_OfinformationGain = informationGain;
					max_Threshold = threshold[id_th];
					indexOfmaxDimension = dim;
				}

			}
			dim++;
		}
		dimIndex[idNode] = indexOfmaxDimension;
		thresholds[idNode] = max_Threshold;
		splitData(indexOfmaxDimension, idNode, max_Threshold);
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat DensityTree::densityXY()
{
	Mat solution;
	Mat marginal_P = Mat::zeros(numOfsamples, 2, CV_32F);
	Ptr<ml::EM>em_model = ml::EM::create();
	em_model->setClustersNumber(1);
	Mat xy = Mat::zeros(1, 2, CV_32F);
	Vec2d log_Px_iy_i;
	Mat waste;

	for (int leaf_index = 0; leaf_index < leafCount; leaf_index++)
	{
		Mat nodeData;
		Mat not_used;
		nodes[leaf_index + internalCount].convertTo(nodeData, CV_32F); //in unserem Fall internalCount = 0
		bool isTrained = em_model->train(nodeData, 0, not_used);

		if (isTrained)
		{
			for (int x1_index = 0; x1_index < sampleNumber[leaf_index + internalCount]; x1_index++)
			{
				xy.at<float>(0, 0) = nodeData.at<float>(x1_index, 0);

				for (int x2_index = 0; x2_index < sampleNumber[leaf_index + internalCount]; x2_index++)
				{
					xy.at<float>(0, 1) = nodeData.at<float>(x2_index, 1);
					log_Px_iy_i = em_model->predict2(xy, waste);
					float Px_iy_i = -(float)exp(log_Px_iy_i.val[0]);
					//Px_i = Px_i + Px_iy_i
					marginal_P.at<float>(sampleIndex[leaf_index + internalCount][x1_index], 0) += Px_iy_i;
					//Py_i = Py_i + Px_iy_i
					marginal_P.at<float>(sampleIndex[leaf_index + internalCount][x2_index], 1) += Px_iy_i;
				}
			}
		}
	}
	marginal_P.convertTo(solution, CV_64F);
	return solution;
}



