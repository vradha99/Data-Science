#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
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
Mat reducePCA(Mat &dataMatrix, unsigned int dim);
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
	Mat dataPCA = reducePCA(dataMatrix,2);
	Draw2DManifold(dataPCA,"PCA",nSamplesI,nSamplesJ);
	
	// Isomap
	Mat dataIsomap = reduceIsomap(dataMatrix,2);
	Draw2DManifold(dataIsomap,"ISOMAP",nSamplesI,nSamplesJ);
	
	//LLE
	Mat dataLLE = reduceLLE(dataMatrix,2);
	Draw2DManifold(dataLLE,"LLE",nSamplesI,nSamplesJ);
	
	waitKey(0);


	return 0;
}

Mat reducePCA(Mat &dataMatrix, unsigned int dim)
{
	  Mat cov, mu; 
calcCovarMatrix(dataMatrix, cov, mu, 

                      CV_COVAR_NORMAL | CV_COVAR_ROWS); 

	Mat w =  Mat(cov.rows, cov.cols, CV_64F);
	Mat u =  Mat(cov.rows, cov.cols, CV_64F);
	Mat vt =  Mat(cov.rows, cov.cols, CV_64F);
  //cout << "cov: " << endl; 

  //cout << cov << endl; 



  //cout << "mu: " << endl; 

 // cout << mu << endl; 
  //cv::SVD svdMat(cov);
  SVD::compute(cov, w, u, vt);
  cout<<u.size()<<endl;
  Mat eig (u, Rect(0, 0, u.rows-1,u.cols) ); 
  //cout<<eig.size()<<endl;
Mat eig_tp =  Mat(eig.cols, eig.rows, CV_64F);
   Mat dat_tp =  Mat(dataMatrix.cols, dataMatrix.rows, CV_64F);
   cv::transpose(eig, eig_tp);
   cv::transpose(dataMatrix, dat_tp);
 // cout<<eig.cols<<endl;
 // cout<<eig_tp.cols<<endl;
  Mat pca =eig_tp * dat_tp;
   Mat pca_tp =  Mat(pca.cols, pca.rows, CV_64F);
   cv::transpose(pca, pca_tp);
  cout<<pca_tp.size()<<endl;
	
	
	return pca_tp/50;
}

Mat reduceLLE(Mat &dataMatrix, unsigned int dim)
{
	
		   	//Compute the pairwise Euclidean distance matrix.
Mat sq =  Mat(dataMatrix.rows, 3, CV_64F);
	pow(dataMatrix,2,sq);
	 Mat red = Mat::zeros(dataMatrix.rows,1, CV_64F);
	 reduce(sq,red, 1, CV_REDUCE_SUM);
	 
	 Mat rep = repeat(red,1, dataMatrix.rows);
	 
	 
	 Mat dataMatrix_t=Mat(dataMatrix.cols, dataMatrix.rows, CV_64F);
	cv::transpose(dataMatrix, dataMatrix_t);
	Mat rep_t=Mat(rep.cols, rep.rows, CV_64F);
	
	cv::transpose(rep, rep_t);
	
	Mat sqt_i=(rep+rep_t - 2*dataMatrix*dataMatrix_t);
	
	//cout<<sqt_i.size()<<endl;
	//cout<<sqt_i.size()<<endl;
	
	
	Mat sqt_o=Mat(sqt_i.rows, sqt_i.cols, CV_64F);
	sqrt(sqt_i,sqt_o);
	//cout<<sqt_o<<endl;
	   


 int temp;
 Mat hold =  Mat(dataMatrix.rows, dataMatrix.rows, CV_64F,100000.0);
//Number of NN for the graph.


 int k=10;
   
   
   //Compute the k-NN connectivity.
   
   
    Mat sqt_oI;
    
    cv::sortIdx(sqt_o, sqt_oI, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
	cv::sort(sqt_o, sqt_o, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
   
    for (int i = 0; i < dataMatrix.rows; ++i)
	for (int j = 0; j < k; ++j)
     {
		 temp= sqt_oI.at<int>(i,j);
	     hold.at<double>(i,temp)=sqt_o.at<double>(i,j);
	 }
	
	return dataMatrix;
}

Mat reduceIsomap(Mat &dataMatrix, unsigned int dim)
{
	
	

	   	//Compute the pairwise Euclidean distance matrix.
Mat sq =  Mat(dataMatrix.rows, 3, CV_64F);
	pow(dataMatrix,2,sq);
	 Mat red = Mat::zeros(dataMatrix.rows,1, CV_64F);
	 reduce(sq,red, 1, CV_REDUCE_SUM);
	 
	 Mat rep = repeat(red,1, dataMatrix.rows);
	 
	 
	 Mat dataMatrix_t=Mat(dataMatrix.cols, dataMatrix.rows, CV_64F);
	cv::transpose(dataMatrix, dataMatrix_t);
	Mat rep_t=Mat(rep.cols, rep.rows, CV_64F);
	
	cv::transpose(rep, rep_t);
	
	Mat sqt_i=(rep+rep_t - 2*dataMatrix*dataMatrix_t);
	
	//cout<<sqt_i.size()<<endl;
	//cout<<sqt_i.size()<<endl;
	
	
	Mat sqt_o=Mat(sqt_i.rows, sqt_i.cols, CV_64F);
	sqrt(sqt_i,sqt_o);
	//cout<<sqt_o<<endl;
	   


 int temp;
 Mat hold =  Mat(dataMatrix.rows, dataMatrix.rows, CV_64F,100000.0);
//Number of NN for the graph.


 int k=10;
   
   
   //Compute the k-NN connectivity.
   
   
    Mat sqt_oI;
    
    cv::sortIdx(sqt_o, sqt_oI, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
	cv::sort(sqt_o, sqt_o, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
   
    for (int i = 0; i < dataMatrix.rows; ++i)
	for (int j = 0; j < k; ++j)
     {
		 temp= sqt_oI.at<int>(i,j);
	     hold.at<double>(i,temp)=sqt_o.at<double>(i,j);
	 }
	 
	 //using floyed algorithm
	 
	for (int i = 0; i < dataMatrix.rows; ++i)
	for (int j = 0; j < dataMatrix.rows; ++j)  
	for (int k = 0; k < dataMatrix.rows; ++k)
	{
		if((hold.at<double>(j, i)+ hold.at<double>(i, k))< hold.at<double>(j,k))
		{
			hold.at<double>(j,k)= hold.at<double>(j,i)+ hold.at<double>(i,k);
		}  
	} 
    Mat columnOne= Mat::ones(100, 1, CV_64F);

    
    Mat columnOne_t= Mat::zeros(1, 100, CV_64F);
    
    transpose(columnOne, columnOne_t);

    
	Mat i= Mat::eye(100, 100, CV_64F);

	
    
	
	Mat dHold= Mat::ones(hold.rows, hold.cols, CV_64F);
	dHold= hold.mul(hold);


	Mat c= i-(columnOne* columnOne_t)/dataMatrix.rows; 
	Mat centers= Mat::ones(100, 100, CV_64F);
	centers= -0.5* c* dHold* c;
	
	//Eigen vectors and their values
	
	Mat eig_val, eig_vec;
	
	cv::eigen(centers, eig_val, eig_vec);

	Mat v= eig_vec(cv::Rect(0,0,sqt_o.cols, dim)).clone();
	
	
	
	
	  
	return v.t();
	
	

	


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


