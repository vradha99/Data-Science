
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

// functions for drawing
void Draw3DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ);
void Draw2DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ);

// functions for dimensionality reduction
// first parameter: features saved in rows
// second parameter: desired # of dimensions ( here: 2)
//Mat reducePCA(Mat &dataMatrix, unsigned int dim);
Mat reduceIsomap(Mat &dataMatrix, unsigned int dim);
Mat reduceLLE(Mat &dataMatrix, unsigned int dim);

int main(int argc, char** argv)
{
	// generate Data Matrix
	unsigned int nSamplesI = 10;
	unsigned int nSamplesJ = 10;
	Mat dataMatrix =  Mat(nSamplesI*nSamplesJ, 3, CV_64F);
	// noise in the data
	double noiseScaling = 1000.0;

	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			dataMatrix.at<double>(i*nSamplesJ+j,0) = (i/(double)nSamplesI * 2.0 * 3.14 + 3.14) * cos(i/(double)nSamplesI * 2.0 * 3.14) + (rand() % 100)/noiseScaling;
			dataMatrix.at<double>(i*nSamplesJ+j,1) = (i/(double)nSamplesI * 2.0 * 3.14 + 3.14) * sin(i/(double)nSamplesI * 2.0 * 3.14) + (rand() % 100)/noiseScaling;
			dataMatrix.at<double>(i*nSamplesJ+j,2) = 10.0*j/(double)nSamplesJ + (rand() % 100)/noiseScaling;
		}
	}

	// Draw 3D Manifold
	Draw3DManifold(dataMatrix, "3D Points",nSamplesI,nSamplesJ);

	// PCA
	//Mat dataPCA = reducePCA(dataMatrix,2);
	//Draw2DManifold(dataPCA,"PCA",nSamplesI,nSamplesJ);

	// Isomap
	//Mat dataIsomap = reduceIsomap(dataMatrix,2);
	//Draw2DManifold(dataIsomap,"ISOMAP",nSamplesI,nSamplesJ);

	waitKey(0);


	return 0;
}

//Mat reducePCA(Mat &dataMatrix, unsigned int dim)
//{
	//return dataMatrix;
//}

Mat reduceIsomap(Mat &dataMatrix, unsigned int dim)
{
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

	int k = 10;

	//Save the index before sort the Distance-Matrix
	Mat sortedIndex;
	cv::sortIdx(distanceMatrix, sortedIndex, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
	//cout<<"dist matrix"<<distanceMatrix<<endl<<endl;
    //cout<<"sorted"<<sortedIndex.at<double>(0,0)<<endl;
	//Sort the Distance-Matrix
	cv::sort(distanceMatrix, distanceMatrix, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

	Mat kNNdistances = Mat(dataMatrix.rows, dataMatrix.rows, CV_64F, 100000.0);
	int index;

	for (int i = 0; i < dataMatrix.rows; ++i)
		for (int j = 0; j < k; ++j) //Take the values with the correct index in the matrix, for the first k nearest neighbours
		{
			index = sortedIndex.at<int>(i, j);
			kNNdistances.at<double>(i, index) = distanceMatrix.at<double>(i, j); //Save the value on the correct (previous) place
		}


	for (int k = 0; k < kNNdistances.rows; k++)
		for (int i = 0; i < kNNdistances.rows; i++)
			for (int j = 0; j < kNNdistances.rows; j++)
			{
				if (kNNdistances.at<double>(i, j) > (kNNdistances.at<double>(i, k) + kNNdistances.at<double>(k, j)))
				{
					kNNdistances.at<double>(i, j) = kNNdistances.at<double>(i, k) + kNNdistances.at<double>(k, j);
				}
			}

			Mat potence = Mat::ones(kNNdistances.rows, kNNdistances.cols, CV_64F);
			Mat fpot=Mat(kNNdistances.rows,kNNdistances.cols,CV_64F);
			fpot=(1/kNNdistances.rows)*potence;
			cv::Mat R = cv::Mat::eye(kNNdistances.rows, kNNdistances.cols, CV_64F);
			Mat cm=Mat(kNNdistances.rows,kNNdistances.cols,CV_64F);
			cm=R-fpot;
            Mat B=Mat(kNNdistances.rows,kNNdistances.cols,CV_64F);
            B=cm*kNNdistances;
            Mat B_tp=Mat(B.cols,B.rows,CV_64F);
            cv::transpose(B, B_tp);
            B=(B*B_tp);
            Mat A=Mat(kNNdistances.rows,kNNdistances.cols,CV_64F);
            A=(-0.5*B);

   Mat eig_val, eig_vec;

	//Eigenvector-Problem
	cv::eigen(A, eig_val, eig_vec);

	//cout << eig_val << endl;
	//cout << eig_vec << endl;

	//Scaling in 2-dim space
	Mat rectangle = eig_vec(cv::Rect(0, 0, distanceMatrix.cols, dim)).clone();
	Mat result = rectangle.t();
	return result;

}

void Draw3DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ)
{
	Mat origImage = Mat(1000,1000,CV_8UC3);
	origImage.setTo(0.0);
	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			Point p1;
			p1.x = dataMatrix.at<double>(i*nSamplesJ+j,0)*50.0 +500.0 - dataMatrix.at<double>(i*nSamplesJ+j,2)*10;
			p1.y = dataMatrix.at<double>(i*nSamplesJ+j,1) *50.0 + 500.0 - dataMatrix.at<double>(i*nSamplesJ+j,2)*10;
			circle(origImage,p1,3,Scalar( 255, 255, 255 ));

			Point p2;
			if(i < nSamplesI-1)
			{
				p2.x = dataMatrix.at<double>((i+1)*nSamplesJ+j,0)*50.0 +500.0 - dataMatrix.at<double>((i+1)*nSamplesJ+(j),2)*10;
				p2.y = dataMatrix.at<double>((i+1)*nSamplesJ+j,1) *50.0 + 500.0 - dataMatrix.at<double>((i+1)*nSamplesJ+(j),2)*10;

				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
			if(j < nSamplesJ-1)
			{
				p2.x = dataMatrix.at<double>((i)*nSamplesJ+j+1,0)*50.0 +500.0 - dataMatrix.at<double>((i)*nSamplesJ+(j+1),2)*10;
				p2.y = dataMatrix.at<double>((i)*nSamplesJ+j+1,1) *50.0 + 500.0 - dataMatrix.at<double>((i)*nSamplesJ+(j+1),2)*10;

				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
		}
	}


	namedWindow( name, WINDOW_AUTOSIZE );
	imshow( name, origImage );
}

void Draw2DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ)
{
	Mat origImage = Mat(1000,1000,CV_8UC3);
	origImage.setTo(0.0);
	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			Point p1;
			p1.x = dataMatrix.at<double>(i*nSamplesJ+j,0)*1000.0 +500.0;
			p1.y = dataMatrix.at<double>(i*nSamplesJ+j,1) *1000.0 + 500.0;
			//circle(origImage,p1,3,Scalar( 255, 255, 255 ));

			Point p2;
			if(i < nSamplesI-1)
			{
				p2.x = dataMatrix.at<double>((i+1)*nSamplesJ+j,0)*1000.0 +500.0;
				p2.y = dataMatrix.at<double>((i+1)*nSamplesJ+j,1) *1000.0 + 500.0;
				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
			if(j < nSamplesJ-1)
			{
				p2.x = dataMatrix.at<double>((i)*nSamplesJ+j+1,0)*1000.0 +500.0;
				p2.y = dataMatrix.at<double>((i)*nSamplesJ+j+1,1) *1000.0 + 500.0;
				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}

		}
	}


	namedWindow( name, WINDOW_AUTOSIZE );
	imshow( name, origImage );
	imwrite( (String(name) + ".png").c_str(),origImage);
}

