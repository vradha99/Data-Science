/*CV_8U <-> uchar
CV_32S <-> int
CV_32F <-> float
CV_64F <-> double
*/
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

// functions for drawing
void Draw3DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ);
void Draw2DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ);

// functions for dimensionality reduction
// first parameter: features saved in rows
// second parameter: desired (wuenschen) # of dimensions ( here: 2)
Mat reducePCA(Mat &dataMatrix, unsigned int dim);
Mat reduceIsomap(Mat &dataMatrix, unsigned int dim);

int main(int argc, char** argv)
{
	// generate Data Matrix
	unsigned int nSamplesI = 10;
	unsigned int nSamplesJ = 10;
	Mat dataMatrix = Mat(nSamplesI*nSamplesJ, 3, CV_64F);
	// noise in the data
	double noiseScaling = 1000.0;

	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			dataMatrix.at<double>(i*nSamplesJ + j, 0) = (i / (double)nSamplesI * 2.0 * 3.14 + 3.14) * cos(i / (double)nSamplesI * 2.0 * 3.14) + (rand() % 100) / noiseScaling;
			dataMatrix.at<double>(i*nSamplesJ + j, 1) = (i / (double)nSamplesI * 2.0 * 3.14 + 3.14) * sin(i / (double)nSamplesI * 2.0 * 3.14) + (rand() % 100) / noiseScaling;
			dataMatrix.at<double>(i*nSamplesJ + j, 2) = 10.0*j / (double)nSamplesJ + (rand() % 100) / noiseScaling;
		}
	}

	// Draw 3D Manifold
	Draw3DManifold(dataMatrix, "3D Points", nSamplesI, nSamplesJ);

	// PCA
	Mat dataPCA = reducePCA(dataMatrix,2);
	Draw2DManifold(dataPCA,"PCA",nSamplesI,nSamplesJ);

	// Isomap
	Mat dataIsomap = reduceIsomap(dataMatrix,2);

	Draw2DManifold(dataIsomap,"ISOMAP",nSamplesI,nSamplesJ);

	waitKey(0);


	return 0;
}

Mat reducePCA(Mat &dataMatrix, unsigned int dim)
{

      Mat mus=Mat(1,dataMatrix.cols,CV_64F);
      double r1m,r2m,r3m;
      //cout<<"dataMatrix"<<dataMatrix<<endl;
      for(int i =0;i< dataMatrix.rows;++i)
      {
       r1m=r1m+dataMatrix.at<double>(i,0);
       r2m=r2m+dataMatrix.at<double>(i,1);
       r3m=r3m+dataMatrix.at<double>(i,2);
      }


r1m=r1m/dataMatrix.rows;
r2m=r2m/dataMatrix.rows;
r3m=r3m/dataMatrix.rows;
mus.at<double>(0,0)=r1m;
mus.at<double>(0,1)=r2m;
mus.at<double>(0,2)=r3m;
//cout<<"mus:"<<mus<<endl;

Mat zeromMatrix=Mat (dataMatrix.rows,dataMatrix.cols,CV_64F);

  for(int i=0;i<dataMatrix.rows;i++)
{
zeromMatrix.at<double>(i,0)=dataMatrix.at<double>(i,0)-mus.at<double>(0,0);
zeromMatrix.at<double>(i,1)=dataMatrix.at<double>(i,1)-mus.at<double>(0,1);
zeromMatrix.at<double>(i,2)=dataMatrix.at<double>(i,2)-mus.at<double>(0,2);
}

Mat zeromMatrix_tp=Mat(zeromMatrix.cols,zeromMatrix.rows,CV_64F);

 cv::transpose(zeromMatrix, zeromMatrix_tp);
 Mat mycov=Mat(zeromMatrix_tp.rows,zeromMatrix.cols,CV_64F);
 mycov=(zeromMatrix_tp*zeromMatrix);
 //cout<<"my cov"<<mycov<<endl;


	Mat w =  Mat(mycov.rows, mycov.cols, CV_64F);
	Mat u =  Mat(mycov.rows, mycov.cols, CV_64F);
	Mat vt =  Mat(mycov.rows, mycov.cols, CV_64F);

  SVD::compute(mycov, w, u, vt);

  Mat eig (u, Rect(0, 0, u.rows-1,u.cols) );//dimensionality reduction:Eliminate the last row 3->2(rows-1)...
return dataMatrix;

Mat eig_tp =  Mat(eig.cols, eig.rows, CV_64F);
   Mat dat_tp =  Mat(dataMatrix.cols, dataMatrix.rows, CV_64F);
   cv::transpose(eig, eig_tp);
   cv::transpose(dataMatrix, dat_tp);

  Mat pca =eig_tp * dat_tp;
  //cout<<"pcam:"<<pca<<endl;
   Mat pca_tp =  Mat(pca.cols, pca.rows, CV_64F);
   cv::transpose(pca, pca_tp);
return pca_tp/50;

}

Mat reduceIsomap(Mat &dataMatrix, unsigned int dim)
{
	///////////////////////////////////////
	//1)Compute the euclidian distances

	//Euclidian Distance in Matrix
	Mat distanceMatrix = Mat::zeros(dataMatrix.rows, dataMatrix.rows, CV_64F);
	for (int i = 0; i < dataMatrix.rows; i++)
	{
		double dist_x = 0;
		double dist_y = 0;
		double dist_z = 0;
		double sum = 0;

		for (int j = 0; j < dataMatrix.rows; j++)
		{
			//Substract the coordinates and take the power of 2
			dist_x = pow((dataMatrix.at<double>(i, 0) - dataMatrix.at<double>(j, 0)), 2);
			dist_y = pow((dataMatrix.at<double>(i, 1) - dataMatrix.at<double>(j, 1)), 2);
			dist_z = pow((dataMatrix.at<double>(i, 2) - dataMatrix.at<double>(j, 2)), 2);
			sum = dist_x + dist_y + dist_z;
			//Sqrt in DistanceMatrix
			distanceMatrix.at<double>(i, j) = sqrt(sum);
		}
	}
	//cout << distanceMatrix << endl;

	///////////////////////////////////////
	//2)Compute the neighbours and Construct a neighborhood graph

	//Number of Neighbours
	int k = 10;

	//Save the index before sort the Distance-Matrix
	Mat sortedIndex;
	cv::sortIdx(distanceMatrix, sortedIndex, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
	//cout<<"dist matrix"<<distanceMatrix<<endl<<endl;
    //cout<<"sorted"<<sortedIndex.at<double>(0,0)<<endl;
	//Sort the Distance-Matrix
	cv::sort(distanceMatrix, distanceMatrix, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);


	//Matrix only with the Distances of the knn, default for the rest = 10000
	Mat kNNdistances = Mat(dataMatrix.rows, dataMatrix.rows, CV_64F, 100000.0);
	int index;

	for (int i = 0; i < dataMatrix.rows; ++i)
		for (int j = 0; j < k; ++j) //Take the values with the correct index in the matrix, for the first k nearest neighbours
		{
			index = sortedIndex.at<int>(i, j);
			kNNdistances.at<double>(i, index) = distanceMatrix.at<double>(i, j); //Save the value on the correct (previous) place
		}



	///////////////////////////////////////
	//3)Compute shortest path between two nodes via Floyd-Warshall algorithm

	for (int k = 0; k < kNNdistances.rows; k++)
		for (int i = 0; i < kNNdistances.rows; i++)
			for (int j = 0; j < kNNdistances.rows; j++)
			{
				if (kNNdistances.at<double>(i, j) > (kNNdistances.at<double>(i, k) + kNNdistances.at<double>(k, j)))
				{
					kNNdistances.at<double>(i, j) = kNNdistances.at<double>(i, k) + kNNdistances.at<double>(k, j);
				}
			}

	///////////////////////////////////////
	//4)Compute lower-dimensional embedding via Multidimensional scaling


	//Create Matrix A https://de.wikipedia.org/wiki/Multidimensionale_Skalierung
	Mat potence = Mat::ones(kNNdistances.rows, kNNdistances.cols, CV_64F);
    cout<<"pot"<<potence<<endl;
	potence = kNNdistances.mul(kNNdistances);

	//cout<<"knndis"<<kNNdistances<<endl;

	Mat matrixA = Mat(kNNdistances.rows, kNNdistances.cols, CV_64F);
	matrixA = (-0.5 * potence);
	//cout << matrixA << endl;

	//Average --> a_i.
	Mat avgColsA = Mat(matrixA.rows, 1, CV_64F);
	for (int i = 0; i < matrixA.rows; i++)
	{
		double sum = 0;
		for (int j = 0; j < matrixA.cols; j++)
		{
			sum += matrixA.at<double>(i, j);
		}
		avgColsA.at<double>(i) = sum / matrixA.cols;
	}
	//cout << sumColsA << endl;

	//Average --> a_.j
	Mat avgRowsA = Mat(1, matrixA.cols, CV_64F);
	for (int j = 0; j < matrixA.cols; j++)
	{
		double sum = 0;
		for (int i = 0; i < matrixA.rows; i++)
		{
			sum += matrixA.at<double>(i, j);
		}
		avgRowsA.at<double>(j) = sum / matrixA.rows;
	}
	//cout << sumRowsA << endl;

	//Average --> a_ij
	double sumA = 0;
	for (int i = 0; i < matrixA.rows; i++)
	{
		for (int j = 0; j < matrixA.cols; j++)
		{
			sumA += matrixA.at<double>(i, j);
		}
	}
	double avgElementsA = sumA / (matrixA.rows*matrixA.cols);

	//Create Matrix B https://de.wikipedia.org/wiki/Multidimensionale_Skalierung
	Mat matrixB = Mat(matrixA.rows, matrixA.cols, CV_64F);
	for (int i = 0; i < matrixA.rows; i++)
	{
		for (int j = 0; j < matrixA.cols; j++)
		{
			matrixB.at<double>(i, j) = matrixA.at<double>(i, j) - avgColsA.at<double>(i) - avgRowsA.at<double>(j) + avgElementsA;
		}
	}

	//Get the Eigenvector and Eigenvalues
	Mat eig_val, eig_vec;

	//Eigenvector-Problem
	cv::eigen(matrixB, eig_val, eig_vec);

	//cout << eig_val << endl;
	//cout << eig_vec << endl;

	//Scaling in 2-dim space
	Mat rectangle = eig_vec(cv::Rect(0, 0, distanceMatrix.cols, dim)).clone();
	Mat result = rectangle.t();
	return result;
}

void Draw3DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ)
{
	Mat origImage = Mat(1000, 1000, CV_8UC3);
	origImage.setTo(0.0);
	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			Point p1;
			p1.x = dataMatrix.at<double>(i*nSamplesJ + j, 0)*50.0 + 500.0 - dataMatrix.at<double>(i*nSamplesJ + j, 2) * 10;
			p1.y = dataMatrix.at<double>(i*nSamplesJ + j, 1) *50.0 + 500.0 - dataMatrix.at<double>(i*nSamplesJ + j, 2) * 10;
			circle(origImage, p1, 3, Scalar(255, 255, 255));

			Point p2;
			if (i < nSamplesI - 1)
			{
				p2.x = dataMatrix.at<double>((i + 1)*nSamplesJ + j, 0)*50.0 + 500.0 - dataMatrix.at<double>((i + 1)*nSamplesJ + (j), 2) * 10;
				p2.y = dataMatrix.at<double>((i + 1)*nSamplesJ + j, 1) *50.0 + 500.0 - dataMatrix.at<double>((i + 1)*nSamplesJ + (j), 2) * 10;

				line(origImage, p1, p2, Scalar(255, 255, 255), 1, 8);
			}
			if (j < nSamplesJ - 1)
			{
				p2.x = dataMatrix.at<double>((i)*nSamplesJ + j + 1, 0)*50.0 + 500.0 - dataMatrix.at<double>((i)*nSamplesJ + (j + 1), 2) * 10;
				p2.y = dataMatrix.at<double>((i)*nSamplesJ + j + 1, 1) *50.0 + 500.0 - dataMatrix.at<double>((i)*nSamplesJ + (j + 1), 2) * 10;

				line(origImage, p1, p2, Scalar(255, 255, 255), 1, 8);
			}
		}
	}
	namedWindow(name, WINDOW_AUTOSIZE);
	imshow(name, origImage);
	imwrite("3d.png", origImage);
}

void Draw2DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ)
{
	Mat origImage = Mat(1000, 1000, CV_8UC3);
	origImage.setTo(0.0);
	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			Point p1;
			p1.x = dataMatrix.at<double>(i*nSamplesJ + j, 0)*1000.0 + 500.0;
			p1.y = dataMatrix.at<double>(i*nSamplesJ + j, 1) *1000.0 + 500.0;
			//circle(origImage,p1,3,Scalar( 255, 255, 255 ));

			Point p2;
			if (i < nSamplesI - 1)
			{
				p2.x = dataMatrix.at<double>((i + 1)*nSamplesJ + j, 0)*1000.0 + 500.0;
				p2.y = dataMatrix.at<double>((i + 1)*nSamplesJ + j, 1) *1000.0 + 500.0;
				line(origImage, p1, p2, Scalar(255, 255, 255), 1, 8);
			}
			if (j < nSamplesJ - 1)
			{
				p2.x = dataMatrix.at<double>((i)*nSamplesJ + j + 1, 0)*1000.0 + 500.0;
				p2.y = dataMatrix.at<double>((i)*nSamplesJ + j + 1, 1) *1000.0 + 500.0;
				line(origImage, p1, p2, Scalar(255, 255, 255), 1, 8);
			}

		}
	}


	namedWindow(name, WINDOW_AUTOSIZE);
	imshow(name, origImage);
	imwrite((String(name) + ".png").c_str(), origImage);
}


/*
//////////////////////////////////////////////
Power of 2 --> Sum --> Substract --> Sqrt

//Euclidian Distance
Mat potenceValues = Mat(dataMatrix.rows, 3, CV_64F);
//Write power of 2 in potence
pow(dataMatrix, 2, potenceValues);
//cout << dataMatrix << endl;
//Create Matrix with sum of all Columns
Mat sumMatrix = Mat::zeros(dataMatrix.rows, 1, CV_64F);
reduce(potenceValues, sumMatrix, 1, CV_REDUCE_SUM);

//Transpose dataMatrix
Mat dataMatrix_t = Mat(dataMatrix.cols, dataMatrix.rows, CV_64F);
cv::transpose(dataMatrix, dataMatrix_t);

//Transpose the sumMatrix
Mat repeatedSumMatrix = repeat(sumMatrix, 1, dataMatrix.rows);
Mat repeatedSumMatrix_t = Mat(repeatedSumMatrix.cols, repeatedSumMatrix.rows, CV_64F);
cv::transpose(repeatedSumMatrix, repeatedSumMatrix_t);

/*Mat sqt = Mat(repeatedSumMatrix.cols, repeatedSumMatrix.rows, CV_64F);

for (int i = 0; i < sqt.rows; i++)
{
for (int j = 0; j < sqt.cols; j++)
{
sqt.at<double>(i, j) = sumMatrix.at<double>(i) - sumMatrix.at<double>(j);
}
}

//Save alle Distances from all Points
//|x-y|^2 = (x-y)^T (x-y) = -2 * x^T y + x^T x + y^T y
Mat sqt = (-2 * dataMatrix*dataMatrix_t + repeatedSumMatrix + repeatedSumMatrix_t);

//Take the Squareroot
Mat distanceMatrix = Mat(sqt.rows, sqt.cols, CV_64F);
sqrt(sqt, distanceMatrix);

//Remove the -nan(ind) Values in the Diagonal --> all has to be zero
for (int i = 0; i < distanceMatrix.rows; i++)
{
	distanceMatrix.at<double>(i, i) = 0.0;
}

/////////////////////////////////////////////////////////////
Substract --> Power of 2 --> Sum --> Sqrt

//Euclidian Distance in Matrix
Mat distanceMatrix = Mat::zeros(dataMatrix.rows, dataMatrix.rows, CV_64F);
for (int i = 0; i < dataMatrix.rows; i++)
{
double dist_x = 0;
double dist_y = 0;
double dist_z = 0;
double sum = 0;

for (int j = 0; j < dataMatrix.rows; j++)
{
//Substract the coordinates and take the power of 2
dist_x = pow((dataMatrix.at<double>(i, 0) - dataMatrix.at<double>(j, 0)), 2);
dist_y = pow((dataMatrix.at<double>(i, 1) - dataMatrix.at<double>(j, 1)), 2);
dist_z = pow((dataMatrix.at<double>(i, 2) - dataMatrix.at<double>(j, 2)), 2);
sum = dist_x + dist_y + dist_z;
//Sqrt in DistanceMatrix
distanceMatrix.at<double>(i, j) = sqrt(sum);
}
}
//cout << distanceMatrix << endl;



////////////////////////////////////////////////
Multidimensional Scaling (old)

Mat columnOne = Mat::ones(100, 1, CV_64F);
//cout << columnOne << endl;
Mat columnOne_t = Mat::zeros(1, 100, CV_64F);
//cout << columnOne_t << endl;
transpose(columnOne, columnOne_t);
//cout << columnOne_t << endl;

//identity matrix
Mat identity = Mat::eye(100, 100, CV_64F);

//cout << identity << endl;

Mat dHold = Mat::ones(kNNdistances.rows, kNNdistances.cols, CV_64F);
dHold = kNNdistances.mul(kNNdistances);
cout << dHold << endl;

Mat c = identity - (columnOne* columnOne_t) / dataMatrix.rows;
// 0er = -,01 1er = 0,99
cout << c << endl;
cout << columnOne << endl;
cout << columnOne_t << endl;
Mat centers = Mat::ones(100, 100, CV_64F);
centers = -0.5* c* dHold* c;
cout << centers << endl;

//Eigen vectors and their values
Mat eig_val, eig_vec;
cv::eigen(centers, eig_val, eig_vec);
Mat v = eig_vec(cv::Rect(0, 0, distanceMatrix.cols, dim)).clone();
return v.t();*/
