/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DensityTree.h
 * Author: dalia
 *
 * Created on June 19, 2017, 12:30 AM
 */
 #ifndef DENSITYTREE_H
#define DENSITYTREE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

//using namespace cv;
//using namespace std;



class DensityTree 
{
public:
    DensityTree();
    DensityTree( int D, unsigned int R, cv::Mat X);
    void train();
    cv::Mat densityXY();
    void treeTraining(cv::Mat& x,std::vector<int>& xvec,std::vector<int>& yvec);

private:
     int D;
     int resD;
    unsigned int n_thresholds;
    cv::Mat X;
    bool toggleDim;
    std::vector <cv::Mat>nodes;
    std::vector <cv::Mat>densities;
    cv::Mat leftBranch,rightBranch;
};

#endif /* DENSITYTREE_H */

