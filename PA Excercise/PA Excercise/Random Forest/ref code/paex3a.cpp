#include "DensityTree.h"
#include <iostream>


#include <math.h>
#include<vector>
# define PI 3.14159265358979323846

using namespace cv;
using namespace std;
DensityTree::DensityTree( int D, unsigned int n_thresholds, Mat X)
{
    this-> D=D;
    this-> X=X;
    this-> n_thresholds=n_thresholds;
    this-> toggleDim=false;
    this-> resD=D-1;
}

double entropy(Mat &s)
{
	Mat cov, mu;
	cv::calcCovarMatrix(s, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	double det_cov = determinant(cov);

	return std::log(det_cov);


}

void DensityTree::treeTraining(Mat& x,vector<int>& xvec,vector<int>& yvec)
{
	Mat xCopy=x;
	//cout<<"switch"<<switchDim<<endl;
Mat left_node=Mat::zeros(xCopy.rows, xCopy.cols, CV_64F);
	Mat right_node=Mat::zeros(xCopy.rows, xCopy.cols, CV_64F);

	Mat prev_left_node=Mat::zeros(xCopy.rows, xCopy.cols, CV_64F);
	Mat prev_right_node=Mat::zeros(xCopy.rows, xCopy.cols, CV_64F);
	unsigned int dim;
	vector<int>vec(50,0);
	double cur_ig,prev_ig=-1000,ig;

	for (int d=0; d!= D; ++d) //check if the samples goes to L or R on each dimension //option
		{
		dim=d;
		if (d==0)
		vec=xvec;
		else
		vec=yvec;
	for (unsigned int partition = 0; partition != vec.size(); ++partition)

	{
		Mat cur_left_node=Mat::zeros(xCopy.rows, xCopy.cols, CV_64F);
		Mat cur_right_node=Mat::zeros(xCopy.rows, xCopy.cols, CV_64F);

		unsigned int number_of_left_nodes_d1 = 0;
		unsigned int number_of_right_nodes_d1 = 0;

		for (unsigned int i = 0; i != xCopy.rows; ++i)
		{

			if (xCopy.at<double>(i, dim) < vec[partition])
			{
				cur_left_node.at<double>(number_of_left_nodes_d1,0) = xCopy.at<double>(i,0);
				cur_left_node.at<double>(number_of_left_nodes_d1,1) = xCopy.at<double>(i,1);
				number_of_left_nodes_d1++;
			}
			else
			{
				cur_right_node.at<double>(number_of_right_nodes_d1,0) = xCopy.at<double>(i,0);
				cur_right_node.at<double>(number_of_right_nodes_d1,1) = xCopy.at<double>(i,1);
				number_of_right_nodes_d1++;
			}
		}
		Mat cur_left_node_trunc (cur_left_node, Rect(0, 0, cur_left_node.cols,number_of_left_nodes_d1) );
		Mat cur_right_node_trunc (cur_right_node, Rect(0, 0,cur_right_node.cols,number_of_right_nodes_d1) );
		//cout<<"cur_left_node_trunc "<<cur_left_node_trunc.size()<<endl;
		//cout<<"cur_right_node_trunc "<<cur_right_node_trunc.size()<<endl;


		cur_ig=entropy(xCopy)-(((static_cast<double>(cur_left_node_trunc.rows)/xCopy.rows)*entropy(cur_left_node_trunc))+((static_cast<double>(cur_right_node_trunc.rows)/xCopy.rows)*entropy(cur_right_node_trunc)));

		if(cur_ig>1000)
		cur_ig=-1000;

		if (cur_ig>prev_ig)
		{
		ig=cur_ig;
		left_node=cur_left_node_trunc;
		right_node=cur_right_node_trunc;
		prev_ig=cur_ig;
		prev_left_node=cur_left_node_trunc;
		prev_right_node=cur_right_node_trunc;

		}
		else
		{
		ig=prev_ig;
		left_node=prev_left_node;
		right_node=prev_right_node;
		}
	}
}
	/*
	Mat cov, mu;
	cv::calcCovarMatrix(left_node, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	double det_cov = determinant(cov);
	Mat rcov, rmu;
	cv::calcCovarMatrix(right_node, rcov, rmu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	double rdet_cov = determinant(rcov);
	cout<<"left cov"<<det_cov<<endl;
	cout<<"rcov ="<<cov<<endl;
	cout<<""
	cout<<"right cov"<<rdet_cov<<endl;
	*/

	//vconcat(left_node, right_node, X);
	/* ///// recursion /////////////
	toggleDim= !toggleDim;

	D--;
	if(D>1)
	{

			vector<double> min (2, 0);
	vector<double> max (2,0);
	vector<int> t_x1(50,0);
	vector<int> t_x2(50,0);




	for (int i = 0; i < 2; i++)// find the min and max value in the set of points
	{
	cv::minMaxLoc(X.col(i), &min[i], &max[i]);
	}

	RNG rng(getTickCount());


	for (unsigned int i = 0; i != n_thresholds; ++i) //create random threshlds on the x axis
	{
	t_x1[i] = rng.uniform(min[0],max[0]);

	}

    for (unsigned int j=0; j!= n_thresholds; ++j)//create random values on the y axis
	{
	t_x2[j]=rng.uniform(min[1],max[1]);

}		cout<<"GG"<<endl;
		treeTraining(left_node,t_x1,t_x2);
		for(int i=0;i!=(resD);++i)
		treeTraining(right_node,t_x1,t_x2);

	}
	*/
	//leftBranch=left_node
		//rightBranch=right_node

		nodes.push_back(left_node);
		nodes.push_back(right_node);
}

void DensityTree::train()
{
	vector<double> min (2, 0);
	vector<double> max (2,0);
	vector<int> t_x1(50,0);
	vector<int> t_x2(50,0);


	for (int i = 0; i < 2; i++)// find the min and max value in the set of points
	{
	cv::minMaxLoc(X.col(i), &min[i], &max[i]);
	}

	RNG rng(getTickCount());


	for (unsigned int i = 0; i != n_thresholds; ++i) //create random threshlds on the x axis
	{
	t_x1[i] = rng.uniform(min[0],max[0]);

	}

    for (unsigned int j=0; j!= n_thresholds; ++j)//create random values on the y axis
	{
	t_x2[j]=rng.uniform(min[1],max[1]);

}

//cout<<"switch"<<toggleDim<<endl;
treeTraining(X,t_x1,t_x2);
	//cout<<"switch"<<toggleDim<<endl;
	//std::vector<Mat> S(t_x1.size(),Mat::zeros(X.rows, X.cols, CV_64F));//Mat::zeros(X.rows, X.cols, CV_64F)

	//auto it=S.begin();
	//S.insert(it,X);


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

Mat DensityTree::densityXY()
{


    vector<cv::Mat>  covar(nodes.size());
    vector<cv::Mat>  mean(nodes.size());

    for(unsigned int i=0;i!=nodes.size();++i)
    {
    calcCovarMatrix(nodes[i], covar[i], mean[i],

                      CV_COVAR_NORMAL | CV_COVAR_ROWS);
    //cout<<"covar"<<covar[i]<<endl;
    //cout<<"mean"<<mean[i]<<endl;
}
    Mat j_pdf_left=jointPdf(covar[0],mean[0],nodes[0]);
    Mat j_pdf_right=jointPdf(covar[1],mean[1],nodes[1]);
    Mat m_pdf_left=marginalPdf(j_pdf_left,2);
    Mat m_pdf_right=marginalPdf(j_pdf_right,2);
    //cv::Mat A = (cv::Mat_<float>(2,2) << 1, 2, 3,4);
//cv::Mat B = (cv::Mat_<float>(2,2) << 1, 0, 5, 8);

Mat data;
    vconcat(m_pdf_left, m_pdf_right, data);

    return data;//Temporal
}





