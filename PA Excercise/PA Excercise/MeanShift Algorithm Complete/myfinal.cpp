#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <math.h>
using namespace cv;
using namespace std;
Mat Meanshift(Mat, Mat);

int main()
{
	//Load the image
	//Mat rdimg;
	Mat rdimg = imread("cameraman_noisy.png", CV_LOAD_IMAGE_GRAYSCALE);
	namedWindow("image", WINDOW_NORMAL);
	imshow("image", rdimg);
	waitKey(2000);

	//Reduce the image
	//resize(img, rdimg, Size(64, 64));
	//imshow("Display window", rdimg);
	//waitKey(2000);

	//Create feature space
	double pixelCount = rdimg.cols * rdimg.rows;
	Mat features = Mat(pixelCount, 3, CV_64F, cvScalar(0.));

	//Create the matrix with all features
	int p = 0;
	for (double x = 0.0; x < rdimg.rows; x++)
	{
		for (double y = 0.0; y < rdimg.cols; y++)
		{
			double color = (double)rdimg.at<uchar>(x, y) / 255.0;

			double i = x / rdimg.rows;
			double j = y / rdimg.cols;
			Mat pixel = (Mat_<double>(1, 3) << i, j, color);
			features.row(p) += pixel;
			p++;
		}
	}

	Mat smoothed = Mat(pixelCount, 3, CV_64F, cvScalar(0.));
	Mat current = Mat(1, 3, CV_64F, cvScalar(0.));

	//Compute the meanshifts and put it in the smoorthed-matrix
	for (double p = 0; p < pixelCount; p++)
	{
		features.row(p).copyTo(current);
		current = Meanshift(features, current);
		current.copyTo(smoothed.row(p));
	}

	//Create DenoisedImage to put the smoothed-values in a new image
	Mat denoisedImage(rdimg.rows, rdimg.cols, rdimg.type());
	int index = 0;
	for (int i = 0; i < denoisedImage.rows; i++)
	{
		for (int j = 0; j < denoisedImage.cols; j++)
		{
			denoisedImage.at<uchar>(i, j) = (uchar)(smoothed.at<double>(index, 2) * 255.0);//(int)(smoothed.at<double>(index, 2) * 255.0);
			index++;
		}
	}

	//Show and wirte the denoised Image
	namedWindow("result", WINDOW_NORMAL);
	imshow("result", denoisedImage);
	waitKey(2000);
	imwrite("result.png", denoisedImage);
	waitKey(2000);
	return 0;
}

Mat Meanshift(Mat features, Mat current)
{
	//Initialize values
	double lambda = 0.3;
	int maxIter = 1;
	int scale = 1;
	double tolX = 0.001;
	Mat meanShift = Mat(1, 3, CV_64F, cvScalar(0.));
	int inliers_count = 0;
	int i = 0;

	do
	{
		//substract the current entry from all features entries and calculate the L2norm
		double pixelCount = features.rows;
		Mat featureEuclidian = Mat(pixelCount, 3, CV_64F, cvScalar(0.));
		Mat featuresOneDim = Mat(pixelCount, 1, CV_64F, cvScalar(0.));
		for (double p = 0; p < pixelCount; p++)
		{
			//substract and power of 2
			featureEuclidian.at<double>(p, 0) = std::pow((features.at<double>(p, 0) - current.at<double>(0, 0)), 2.0);
			featureEuclidian.at<double>(p, 1) = std::pow((features.at<double>(p, 1) - current.at<double>(0, 1)), 2.0);
			featureEuclidian.at<double>(p, 2) = std::pow((features.at<double>(p, 2) - current.at<double>(0, 2)), 2.0);

			//add the values of each row and take the squareroot
			featuresOneDim.at<double>(p) = sqrt(featureEuclidian.at<double>(p, 0) + featureEuclidian.at<double>(p, 1) + featureEuclidian.at<double>(p, 2));
		}

		//Check for Inliers
		for (double p = 0; p < pixelCount; p++)
		{
			if (featuresOneDim.at<double>(p) < lambda) //count entries smaller than lambda
			{
				inliers_count++;
				meanShift += features.row(p);
			}
		}

		if (inliers_count != 0)
		{
			//Set the meanshift in current
			meanShift = meanShift / inliers_count;
			meanShift -= current;
			current += meanShift;
			inliers_count = 0;
		}
		else
		{
			return current;
		}
		if (cv::norm(meanShift) < tolX)
		{
			return current;
		}
		i++;
	} while (i <= maxIter);
	return current;
}







/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//int index = 4095;
//for (int i = denoisedImage.rows -1; i >= (denoisedImage.rows * (2/3)); i--)
//{
//	for (int j = denoisedImage.cols - 1; j >= (denoisedImage.rows * (2 / 3)); j--)
//	{
//		denoisedImage.at<uchar>(i, j) = (uchar)(smoothed.at<double>(index, 2) * 255.0);//(int)(smoothed.at<double>(index, 2) * 255.0);
//		index--;
//	}
//}

//for (int p = 0; p < 4096; p++)
//{
//	int x = (int)(p / 64);
//	int y = p % 64;
//	double color = smoothed.at<double>(p, 2) * 255.0;
//	denoisedImage.at<uchar>(x, y) = (uchar)(color);
//}

/*for (int i = 0; i < rdimg.rows; i++) {
uchar* data = denoisedImage.ptr<uchar>(i);
for (int j = 0; j < rdimg.cols; j++) {
data[j] = (uchar)(0);
index++;
}
}*/

//stored the integervalues of smoothed
/*std::ofstream out("denoised.txt");

for (int r = -1; r < 64; r++)
{
if (r == -1) { out << '\t'; }
else if (r >= 0) { out << r << '\t'; }

for (int c = -1; c < 64; c++)
{
if (r == -1 && c >= 0) { out << c << '\t'; }
else if (r >= 0 && c >= 0)
{
out << static_cast<int>(denoisedImage.at<unsigned char>(r, c)) << '\t';
}
}
out << std::endl;
}

//stored the integervalues of the original
std::ofstream out2("Start.txt");

for (int r = -1; r < 64; r++)
{
if (r == -1) { out2 << '\t'; }
else if (r >= 0) { out2 << r << '\t'; }

for (int c = -1; c < 64; c++)
{
if (r == -1 && c >= 0) { out2 << c << '\t'; }
else if (r >= 0 && c >= 0)
{
out2 << static_cast<int>(rdimg.at<unsigned char>(r, c)) << '\t';
}
}
out2 << std::endl;
}*/

/*Mat storeMatrix = Mat(rdimg.rows, rdimg.cols, CV_64F, cvScalar(0.));

for (int i = 0; i < rdimg.rows; i++)
{
for (int j = 0; j < rdimg.cols; j++)
{
double color = smoothed.at<double>(index, 2) * 255.0;
storeMatrix.row(i).col(j) = color;
index++;
}
}

//Stored in Matrix to see values
cout << "Matrix = " << endl << " " << storeMatrix << endl << endl;*/


