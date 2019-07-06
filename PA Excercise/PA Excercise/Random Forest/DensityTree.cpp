/*
 *
 * Compilation line:
 g++ -o main main.cpp DensityTree.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml
 https://www.nowpublishers.com/article/Details/CGV-035 <-density forest paper
*
*/

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
#include<vector>
# define PI 3.14159265358979323846
using namespace cv;
using namespace std;
int numOfNodes;
int numOfLeafs;
int numOfInternal;
Mat marginalPdf(Mat& jClusPdf, unsigned int dim);
Mat jointPdf(Mat& cov, Mat& mu, Mat& clus);
void plotData2(Mat dataMatrix, char const * name);

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

}


double entropy(Mat &s)
{
	Mat cov, mu;
	cv::calcCovarMatrix(s, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	double det_cov = determinant(cov);

	return 0.5* std::log(std::pow(2 * PI* 2.71828, 2) *det_cov);


}
void DensityTree::train()
{
    vector<double> min (2, 0);
	vector<double> max (2,0);
	vector<int> t_x1(50,0);
	vector<int> t_x2(50,0);
    for (int i = 0; i < 2; i++)// find the min and max value in the set of points
	cv::minMaxLoc(X.col(i), &min[i], &max[i]);
    RNG rng(getTickCount());
	for (unsigned int i = 0; i != n_thresholds; ++i) //create random threshlds on the x axis
	t_x1[i] = rng.uniform(min[0],max[0]);
    for (unsigned int j=0; j!= n_thresholds; ++j)//create random values on the y axis
	t_x2[j]=rng.uniform(min[1],max[1]);
	treeTraining(X,t_x1,t_x2);
}


void DensityTree::treeTraining(Mat& x,vector<int>& x1,vector<int>& x2)
{

Mat xc=x;
Mat ln=Mat::zeros(xc.rows, xc.cols, CV_64F);
Mat rn=ln.clone();
Mat pln=ln.clone();
Mat prn=rn.clone();
unsigned int dim;
vector<int>thresholds(n_thresholds,0);

double cur_infg,prev_infg=-1000,infg;
for(int d=0;d!=D;++d)
	 //check if the samples goes to L or R on each dimension //option
		{
		dim=d;
		if (d==0)
		thresholds=x1;
		else
		thresholds=x2;
		for (unsigned int partition = 0; partition != thresholds.size(); ++partition)

		{
		Mat cln=ln.clone();
		Mat crn=rn.clone();

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

}
		nodes.push_back(ln);
		nodes.push_back(rn);
}







Mat DensityTree::densityXY()
{
vector<cv::Mat>  covar(nodes.size());
	vector<cv::Mat>  mean(nodes.size());

		for (int i = 0; i < nodes.size(); i++)
	{
		calcCovarMatrix(nodes[i], covar[i], mean[i], CV_COVAR_NORMAL | CV_COVAR_ROWS);

	}
	  Mat j_pdf_left=jointPdf(covar[0],mean[0],nodes[0]);
    Mat j_pdf_right=jointPdf(covar[1],mean[1],nodes[1]);
    Mat m_pdf_left=marginalPdf(j_pdf_left,2);
    Mat m_pdf_right=marginalPdf(j_pdf_right,2);

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
   // return X;//Temporal, only to not generate an error when compiling
}

Mat jointPdf(Mat& cov,Mat& mu, Mat& clus)
{

	Mat j_pdf=Mat::zeros(clus.rows, clus.rows, CV_64F);
	Mat scalar_val;
	double det_cov = determinant(cov);

	Mat var_vec=Mat::zeros(mu.rows, mu.cols, CV_64F);
	Mat var_vec_t=Mat::zeros(mu.cols, mu.rows, CV_64F);;
	det_cov=pow(det_cov, 0.5);
	Mat cov_inv=cov.inv();
	//var_vec=
	for(int i=0;i != j_pdf.rows;++i)
	{
		for(int k=0;k!=j_pdf.cols;++k)
		{
	var_vec.at<double>(0,0)=clus.at<double>(i,0);
	var_vec.at<double>(0,1)=clus.at<double>(k,1);
	var_vec=var_vec-mu;
	cv::transpose(var_vec, var_vec_t);
	scalar_val= ((var_vec * cov_inv) *(var_vec_t));

	//cout<<(-0.5 * ((var_vec * cov_inv) *(var_vec_t)));
	j_pdf.at<double>(i,k)=1/(2*PI*det_cov) *  std::exp (-0.5 * scalar_val.at<double>(0,0));
	}
	}



	return j_pdf;
	}

Mat marginalPdf (Mat& jClusPdf,unsigned int dim)
{
	Mat m_pdf=Mat::zeros(jClusPdf.rows, dim, CV_64F);

		for(int i=0;i != jClusPdf.rows;++i)
	{
		for(int k=0;k!=jClusPdf.cols;++k)
		{


	m_pdf.at<double>(i,0)+=jClusPdf.at<double>(i,k);
	m_pdf.at<double>(i,1)+=jClusPdf.at<double>(k,i);
	}
	}

	return m_pdf;
	}


