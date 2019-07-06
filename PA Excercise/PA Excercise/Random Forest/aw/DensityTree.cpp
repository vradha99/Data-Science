/*
 *
 * Compilation line:
 g++ -o main main.cpp DensityTree.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml
 https://www.nowpublishers.com/article/Details/CGV-035 <-density forest paper
*
*/
//#include "stdafx.h"
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <math.h>
#include <map>
#include "DensityTree.h.sec"
#include<vector>
# define PI 3.14159265358979323846
using namespace cv;
using namespace std;

//Further Methods
Mat marginalPdf(Mat& jClusPdf, unsigned int dim);
Mat jointPdf(Mat& cov, Mat& mu, Mat& clus);
void plotData2(Mat dataMatrix, char const * name);

//Properties
int numOfNodes;
int numOfLeafs;
int numOfInternal;

DensityTree::DensityTree(unsigned int D, unsigned int n_thresholds, Mat X)
{
	/*
	 * D is the depht of the tree. If D=2, the tree will have 3 nodes, the root and its 2 children.
	 * This is a binari tree. Once you know D, you know the number of total nodes is pow(2,D)-1, the number of leaves or terminal nodes are pow(2,(D-1)).
	 * The left child of the i-th node is the (i*2+1)-th node and the right one is the (i*2+2).
	 * Having this information, you can use simple arrays as a structure to save information of each node. For example, you can save in a boolean vector wheather the
	 * node is a leave or not like this:
	 *
	 * bool *isLeaf=new bool[numOfNodes];
	 * for(int i=0;i<numOfInternal;i++)
	 *     isLeaf[i]=false;
	 * for(int i=numOfInternal;i<numOfNodes;i++)
	 *  isLeaf[i]=true;
	 */
	this->D = D;
	this->X = X;
	this->n_thresholds = n_thresholds;
	numOfNodes = pow(2, D) - 1;
	numOfLeafs = pow(2, (D - 1));
	numOfInternal = numOfNodes - numOfLeafs;
    Sdet = new double [numOfNodes];
	Snum = new unsigned int[numOfNodes];
	Sind = new vector<int> [numOfNodes];
	vector<int> temp;
	for (int i = 0; i < X.rows; i++)
	{
		temp.push_back(i);
	}
	Sind[0] = temp;
	S = new Mat [numOfNodes];

	//first entry = All data and numerator = all samples
	Mat Covar,mu;
	S[0] = X.clone();
	Covar=S[0].clone();
	mu=S[0].clone();
	Snum[0] = X.rows;
	cv::calcCovarMatrix(S[0], Covar, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	Sdet[0] = determinant(Covar);

	//set arrays for thresholds of each internal node and dim of terminals
	thresholds = new double [numOfInternal];
	dimIndex = new unsigned int [numOfNodes];


}

void DensityTree::train()
{
	vector<double> min(2, 0);
	vector<double> max(2,0);
	vector<int> x_thresholds(50,0);
	vector<int> y_thresholds(50,0);


	for (int i = 0; i < 2; i++)// find the min and max value in the set of points
	{
		cv::minMaxLoc(X.col(i), &min[i], &max[i]);
	}

	RNG rng(getTickCount());

	//cout << "min_right" << min[0] << endl;
	//cout << "max_right" << max[0] << endl;

	//cout << "min_left" << min[1] << endl;
	//cout << "max_left" << max[1] << endl;



	for (unsigned int i = 0; i != n_thresholds; ++i) //create random threshlds on the x axis
	{
		x_thresholds[i] = rng.uniform(min[0], max[0]);
		//cout << "treshold_x" << x_thresholds[i] << endl; -> test: different every tree = right!
	}

	for (unsigned int j = 0; j != n_thresholds; ++j)//create random values on the y axis
	{
		y_thresholds[j] = rng.uniform(min[1], max[1]);
		//cout << "treshold_y" << y_thresholds[j] << endl; --> test: different every tree = right!
	}

	training(X, x_thresholds, y_thresholds);
}

/*
Mat DensityTree::densityXY()
{
	vector<cv::Mat>  covar(nodes.size());
	vector<cv::Mat>  mean(nodes.size());

		for (int i = 0; i < nodes.size(); i++)
	{
		//cout << "node_before" << nodes[i] << endl;
		//cout << "covar_before" << covar[i] << endl;
		//cout << "mean_before" << mean[i] << endl;
		calcCovarMatrix(nodes[i], covar[i], mean[i], CV_COVAR_NORMAL | CV_COVAR_ROWS);
		//cout << "node_after" << nodes[i] << endl;
		//cout<<"covar_after"<<covar[i]<<endl;
		//cout<<"mean_after"<<mean[i]<<endl;
	}

	Mat j_pdf_left = jointPdf(covar[0], mean[0], nodes[0]);
	Mat j_pdf_right = jointPdf(covar[1], mean[1], nodes[1]);
	//cout << "m_pdf_right" << j_pdf_left.size << endl;
	//cout << "m_pdf_left" << j_pdf_right.size << endl;
	Mat m_pdf_left = marginalPdf(j_pdf_left, 2);
	Mat m_pdf_right = marginalPdf(j_pdf_right, 2);
	//cv::Mat A = (cv::Mat_<float>(2,2) << 1, 2, 3,4);
	//cv::Mat B = (cv::Mat_<float>(2,2) << 1, 0, 5, 8);

	//cout << "m_pdf_right" << m_pdf_right << endl;
	//cout << "m_pdf_left" << m_pdf_left << endl;

	Mat data;
	vconcat(m_pdf_left, m_pdf_right, data);
	return data;
	/*
	*
	if X=
	[x1_1,x2_1;
	 x1_2,x2_2;
	 ....
	 x1_N,x2_N]

	then you return
	M=
	[Px1,Px2]

	Px1 and Px2 are column vectors of size N (X and M have the same size)
	They are the marginals distributions.
	Check https://en.wikipedia.org/wiki/Marginal_distribution
	Feel free to delete this comments
	Tip: you can use cv::ml::EM::predict2 to estimate the probs of a sample.

	*
	*/

	/*
	////////////////////////////////Test :Only Normal distribution /////////////////////////////////////////////////////
	Mat x_nodes = nodes[0].col(0);
	Mat mean2;
	Mat var;

	meanStdDev(x_nodes, mean2, var);

	cout << "var" << var << endl;
	Mat norm_x_nodes = Mat::zeros(1000, 1, CV_64F);

	for (int i = 0; i < x_nodes.rows; i++)
	{
	double value = x_nodes.at<double>(i);
	norm_x_nodes.at<double>(i) = 1 / (2 * PI*var.at<double>(0)) *  std::exp(-0.5*value*value);
	}

	Mat y_nodes = nodes[0].col(1);
	Mat mean3;
	Mat var2;

	meanStdDev(y_nodes, mean3, var2);
	Mat norm_y_nodes = Mat::zeros(1000, 1, CV_64F);

	for (int i = 0; i < y_nodes.rows; i++)
	{
	double value = y_nodes.at<double>(i);
	norm_y_nodes.at<double>(i) = 1 / (2 * PI*var2.at<double>(0)) *  std::exp(-0.5*value*value);
	}

	Mat left;
	hconcat(norm_x_nodes, norm_y_nodes, left);
	cout << "left" << left.rows << endl;*/


	////////////////////////////////Test: EM-Algorithm --> doesnt work//////////////////////////////
	/*Mat left = nodes[0];
	Mat left_x1 = left.col(0);
	Mat left_x2 = left.col(1);

	Mat probs(left_x1.rows, 2, CV_64FC1);
	Mat results = Mat::zeros(nodes[0].rows, 1, CV_64FC1);
	Ptr<ml::EM> em = ml::EM::create();

	Mat sample = left.row(0);
	cout << "sample" << sample << endl;

	//em->predict2(left, probs);
	//cout << "probs" << probs << endl;

	for (int i = 0; i < nodes[0].rows; i++)
	{
	em->predict2(left_x1.row(i), probs.row(i));
	}
	cout<<"probs"<<probs<<endl;
}
*/

Mat jointPdf(Mat& cov, Mat& mu, Mat& clus)
{

	Mat j_pdf = Mat::zeros(clus.rows, clus.rows, CV_64F);
	Mat scalar_val;
	double det_cov = determinant(cov);

	Mat var_vec = Mat::zeros(mu.rows, mu.cols, CV_64F);
	Mat var_vec_t = Mat::zeros(mu.cols, mu.rows, CV_64F);;
	det_cov = pow(det_cov, 0.5);
	Mat cov_inv = cov.inv();
	//var_vec=
	for (int i = 0; i != j_pdf.rows; ++i)
	{
		for (int k = 0; k != j_pdf.cols; ++k)
		{
			var_vec.at<double>(0, 0) = clus.at<double>(i, 0);
			var_vec.at<double>(0, 1) = clus.at<double>(k, 1);
			var_vec = var_vec - mu;
			cv::transpose(var_vec, var_vec_t);
			scalar_val = ((var_vec * cov_inv) *(var_vec_t));

			//cout<<(-0.5 * ((var_vec * cov_inv) *(var_vec_t)));
			j_pdf.at<double>(i, k) = 1 / (2 * PI*det_cov) *  std::exp(-0.5 * scalar_val.at<double>(0, 0));
		}
	}
	return j_pdf;
}

Mat marginalPdf(Mat& jClusPdf, unsigned int dim)
{
	Mat m_pdf = Mat::zeros(jClusPdf.rows, dim, CV_64F);

	//cout << "jClusPdf" << jClusPdf << endl;

	for (int i = 0; i != jClusPdf.rows; ++i)
	{
		for (int k = 0; k != jClusPdf.cols; ++k)
		{
			m_pdf.at<double>(i, 0) += jClusPdf.at<double>(i, k);
			m_pdf.at<double>(i, 1) += jClusPdf.at<double>(k, i);
		}
	}

	return m_pdf;
}

double entropy(Mat &s)
{
	Mat cov, mu;
	double sum;
	mu=Mat(1,s.cols,CV_64F);
	int col=s.cols;
	int row=s.rows;
	for(int j=0;j<=col;++j)
	{
	for(int i=0;i<=row;++i)
	{
     sum=sum+s.at<double>(i,j);
      }
      sum=sum/s.rows;
      mu.at<double>(0,j)=sum;
     }

     Mat zeromMatrix=Mat (s.rows,s.cols,CV_64F);
      for(int i=0;i<s.rows;i++)
      {
      for(int j=0;j<s.cols;++j)
      {
      zeromMatrix.at<double>(i,j)=s.at<double>(i,j)-mu.at<double>(0,j);
      }
      }
      Mat zeromMatrix_tp=Mat(zeromMatrix.cols,zeromMatrix.rows,CV_64F);

 cv::transpose(zeromMatrix, zeromMatrix_tp);
 Mat mycov=Mat(zeromMatrix_tp.rows,zeromMatrix.cols,CV_64F);
 mycov=(zeromMatrix_tp*zeromMatrix);

	//cv::calcCovarMatrix(s, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	double det_cov = determinant(mycov);

	return 0.5* std::log(std::pow(2 * PI* 2.71828, 2) *det_cov);
}

void DensityTree::training(Mat& x,vector<int>& x1,vector<int>& x2)
{

Mat xc=x;
Mat ln=Mat::zeros(xc.rows, xc.cols, CV_64F);
Mat rn=ln.clone();
Mat pln=ln.clone();
Mat prn=rn.clone();
unsigned int dim;
vector<int>thresholds(n_thresholds,0);

double cur_infg,prev_infg=-1000,infg;
int d=0;
while(d!=D)
	 //check if the samples goes to L or R on each dimension //option
		{
		dim=d;
		if (d==0)
		thresholds=x1;
		else
		thresholds=x2;
		for (unsigned int partition = 0; partition != thresholds.size(); ++partition)

		{
		Mat cln=Mat::zeros(xc.rows, xc.cols, CV_64F);
		Mat crn=Mat::zeros(xc.rows, xc.cols, CV_64F);

			unsigned int numln = 0;
			unsigned int numrn = 0;
           for (unsigned int i = 0; i != xc.rows; ++i)
             if (xc.at<double>(i, dim) < thresholds[partition])
				{
					cln.at<double>(numln, 0) = xc.at<double>(i, 0);
					cln.at<double>(numln, 1) = xc.at<double>(i, 1);
					numln++;
				}
				else
				{
					crn.at<double>(numrn, 0) = xc.at<double>(i, 0);
					crn.at<double>(numrn, 1) = xc.at<double>(i, 1);
					numrn++;
				}
                  Mat clnt(cln, Rect(0, 0, cln.cols, numln));
                  Mat crnt(crn, Rect(0, 0, crn.cols, numrn));



                  cur_infg=entropy(xc)-(((static_cast<double>(clnt.rows)/xc.rows)*entropy(clnt))+((static_cast<double>(crnt.rows)/xc.rows)*entropy(crnt)));

               		if(cur_infg>1000)
		            cur_infg=-1000;

		if (cur_infg>prev_infg)
		{
		infg=cur_infg;
		ln=clnt;
		rn=crnt;
		prev_infg=cur_infg;
		pln=clnt;
		prn=crnt;

		}
		else
		{
		infg=prev_infg;
		ln=pln;
		rn=prn;
		}
	}
	++d;

}
		nodes.push_back(ln);
		nodes.push_back(rn);
}
void DensityTree::split(int dim, int idNode, double threshold)
{
	Mat L = Mat(0, dimensions, CV_64F);
	Mat R = Mat(0, dimensions, CV_64F);

	vector<int> indL;
	vector<int> indR;

	for (int i = 0; i < Snum[idNode]; i++)
	{
		if (S[idNode].at<double>(i, dim) <= threshold)
		{
			indL.push_back(Sind[idNode][i]);
			L.push_back(S[idNode].row(i));
		}
		else
		{
			indR.push_back(Sind[idNode][i]);
			R.push_back(S[idNode].row(i));
		}
	}
	Sind[idNode * 2 + 1] = indL;
	Sind[idNode * 2 + 2] = indR;
	L.copyTo(S[idNode * 2 + 1]);
	R.copyTo(S[idNode * 2 + 2]);
	Snum[idNode * 2 + 1] = S[idNode * 2 + 1].rows;
	Snum[idNode * 2 + 2] = S[idNode * 2 + 2].rows;
	Mat CovL, ML;
	//cout << "S[1]" << S[1] << endl;
	calcCovarMatrix(S[idNode * 2 + 1], CovL, ML, CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_64F);
	Sdet[idNode * 2 + 1] = determinant(CovL);
	Mat CovR, MR;
	//cout << "S[2]" << S[2] << endl;
	calcCovarMatrix(S[idNode * 2 + 2], CovR, MR, CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_64F);
	Sdet[idNode * 2 + 2] = determinant(CovR);

Mat DensityTree::densityXY()
{
	Mat distl;
	Mat dist = Mat::zeros(X.rows, 2, CV_32F);
	Ptr<ml::EM>params = ml::EM::create();
	params->setClustersNumber(1);
	Mat element = Mat::zeros(1, 2, CV_32F);
	Vec2d p;
	Mat t;

	for (int i = 0; i < numOfLeafs; i++)
	{
		Mat tempData;
		Mat response;
		S[i + numOfInternal].convertTo(tempData, CV_32F);

		if (params->train(tempData, 0, response))
		{
			for (int j = 0; j <= Snum[i + numOfInternal]; j++)
			{
				element.at<float>(0, 0) = tempData.at<float>(j, 0);

				for (int k = 0; k < Snum[i + numOfInternal]; k++)
				{
					element.at<float>(0, 1) = tempData.at<float>(k, 1);
					p = params->predict2(element, t);
					float val = (float)exp(p.val[0]);
					dist.at<float>(Sind[i + numOfInternal][j], 0) += val;
					dist.at<float>(Sind[i + numOfInternal][k], 1) += val;

				}
			}
		}
	}
	dist.convertTo(distl, CV_64F);
    return distl;
}









