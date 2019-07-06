// YOUR NAME
// IF NECESSARY: YOUR COMPILATION COMMAND

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

void generateRandomObservations(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount);
double observationProbabilityForward(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount);
double bestStateSequence(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount, unsigned int* bestStates);
double rd() { return (double)rand() / (double)RAND_MAX; } // we suggest to use this to create uniform random values between 0 and 1

int main(int argc, char** argv)
{
	// keep this to produce our results
	srand(42);
	// Four states, 3 symbols
	Mat A =  Mat(4, 4, CV_64F);
	Mat B =  Mat(4, 3, CV_64F);
	Mat P = Mat(4,1, CV_64F);
	A.at<double>(0,0) = 0.5;
	A.at<double>(0,1) = 0.2;
	A.at<double>(0,2) = 0.3;
	A.at<double>(0,3) = 0.0;

	A.at<double>(1,0) = 0.2;
	A.at<double>(1,1) = 0.4;
	A.at<double>(1,2) = 0.1;
	A.at<double>(1,3) = 0.3;

	A.at<double>(2,0) = 0.7;
	A.at<double>(2,1) = 0.1;
	A.at<double>(2,2) = 0.1;
	A.at<double>(2,3) = 0.1;

	A.at<double>(3,0) = 0.0;
	A.at<double>(3,1) = 0.1;
	A.at<double>(3,2) = 0.8;
	A.at<double>(3,3) = 0.1;

	P.at<double>(0,0) = 0.7;
	P.at<double>(0,1) = 0.2;
	P.at<double>(0,2) = 0.1;
	P.at<double>(0,3) = 0.0;

	B.at<double>(0,0) = 0.6;
	B.at<double>(0,1) = 0.2;
	B.at<double>(0,2) = 0.2;

	B.at<double>(1,0) = 0.4;
	B.at<double>(1,1) = 0.4;
	B.at<double>(1,2) = 0.2;

	B.at<double>(2,0) = 0.3;
	B.at<double>(2,1) = 0.3;
	B.at<double>(2,2) = 0.4;

	B.at<double>(3,0) = 0.1;
	B.at<double>(3,1) = 0.2;
	B.at<double>(3,2) = 0.7;

	cout<<"A ="<<A<<endl;

	cout<<"B ="<<B<<endl;

	cout<<"P ="<<P<<endl;
	// Length = 2;
	unsigned int cnt = 2;
	unsigned int obs1[cnt];
	unsigned int bestStates1[cnt];
	generateRandomObservations(A,B,P,obs1,cnt);
	cout << "Observation Sequence: ";
	for( int i = 0; i < cnt; i++)
	{
		cout << obs1[i] << " ";
	}
	cout << endl;
	double prob_all = observationProbabilityForward(A,B,P,obs1,cnt);

	double prob_best = bestStateSequence(A, B, P, obs1, cnt, bestStates1);

	cout << "Probability: " << prob_all << endl;
	cout << "Best Sequence: ";
	for( int i = 0; i < cnt; i++)
	{
		cout << bestStates1[i] << " ";
	}
	cout << "Probability: " << prob_best << endl;
	cout << "Best Prob. / Total Prob: " << prob_best / prob_all << endl << endl;

	// Length = 10
	cnt = 10;
	unsigned int obs2[cnt];
	unsigned int bestStates2[cnt];
	generateRandomObservations(A,B,P,obs2,cnt);
	cout << "Observation Sequence: ";
	for( int i = 0; i < cnt; i++)
	{
		cout << obs2[i] << " ";
	}

	cout << endl;

	prob_all = observationProbabilityForward(A,B,P,obs2,cnt);
	prob_best = bestStateSequence(A, B, P, obs2, cnt, bestStates2);
	cout << "Probability: " << prob_all << endl;
	cout << "Best Sequence: ";
	for( int i = 0; i < cnt; i++)
	{
		cout << bestStates2[i] << " ";
	}
	cout << "Probability: " << prob_best << endl;
	cout << "Best Prob. / Total Prob: " << prob_best / prob_all << endl << endl;
	cout << "Best Prob. - Total Prob: " << prob_best - prob_all << endl << endl;


	return 0;
}

// implement random sampling from the HMM -> Lecture
void generateRandomObservations(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount)
{

		int n_states = A.rows;
		unsigned int states[observationCount];
		cv::Mat cumLikelihoods(B.rows,B.cols,CV_64F);
		for (int i=0;i!=B.rows;++i)
			cumLikelihoods.at<double>(i,0) = B.at<double>(i,0);
		for (int i=0;i!=cumLikelihoods.rows;++i)
			for (int j=1;j!=cumLikelihoods.cols;++j)
				cumLikelihoods.at<double>(i,j) = cumLikelihoods.at<double>(i,j-1) + B.at<double>(i,j);
		cv::Mat cumStates(A.rows,A.cols,CV_64F);
		for (int i=0;i!=cumStates.rows;++i)
			cumStates.at<double>(i,0) = A.at<double>(i,0);
		for (int i=0;i!=cumStates.rows;++i)
			for (int j=1;j!=cumStates.cols;++j)
				cumStates.at<double>(i,j) = cumStates.at<double>(i,j-1) + A.at<double>(i,j);
		cv::Mat cumP(P.rows,P.cols,CV_64F);
		cumP.at<double>(0,0) = P.at<double>(0,0);
		for (int i=1;i!=cumP.cols;++i)
			cumP.at<double>(0,i) = cumP.at<double>(0,i-1) + P.at<double>(0,i);
		double pVal,aVal,bVal;
		pVal = rd();
		int lastState;
		for (int cp=0;cp!=cumP.cols;++cp)
			if (pVal <= cumP.at<double>(0,cp))
			{
				lastState = cp;
				break;
			}
		for (int t=0;t!=observationCount;++t)
		{
			aVal = rd();
			for (int i=0;i!=cumStates.cols;++i)
				if (aVal <= cumStates.at<double>(lastState,i))
				{
					states[t] = i;
					break;
				}

			bVal = rd();
			for (int i=0;i!=cumLikelihoods.cols;++i)
			{
				if (bVal <= cumLikelihoods.at<double>(states[t],i))
				{
					observations[t] = i;
					break;
				}
			}

			lastState = states[t];
		}
}
// Observation probability -> Forward or Backward algorithm
double observationProbabilityForward(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount)
{

		cv::Mat transProb = A.clone();
		cv::Mat likelihoodProb = B.clone();
		cv::Mat intProb = P.clone();
		cv::Mat forMat = cv::Mat(transProb.rows,observationCount,CV_64F);

		for (int i=0;i!=transProb.rows;++i)
		{
			forMat.at<double>(i,0) = intProb.at<double>(0,i)*likelihoodProb.at<double>(i,observations[0]);

		}


		for (unsigned int t=1;t!=observationCount;++t)
		{

			for (int i=0;i!=transProb.rows;++i)
			{
				forMat.at<double>(i,t) = 0;
				for (int j=0;j!=transProb.rows;++j)
					forMat.at<double>(i,t) += forMat.at<double>(i,t-1)*transProb.at<double>(j,i);
				forMat.at<double>(i,t) = forMat.at<double>(i,t) * likelihoodProb.at<double>(i,observations[t]);

			}



		}
		double prob=0;
		for(int i=0;i!=transProb.rows;++i)
		prob+=forMat.at<double>(i,observationCount-1);

	return prob;
}

 void model(cv::Mat& transProb, cv::Mat& likelihoodProb, cv::Mat& intProb)
	{
		double eps = 1e-30;
		for (int i=0;i!=likelihoodProb.rows;++i)
			for (int j=0;j!=likelihoodProb.cols;++j)
				if (likelihoodProb.at<double>(i,j)==0)
					likelihoodProb.at<double>(i,j)=eps;
		for (int i=0;i!=transProb.rows;++i)
			for (int j=0;j!=transProb.cols;++j)
				if (transProb.at<double>(i,j)==0)
					transProb.at<double>(i,j)=eps;
		for (int i=0;i!=intProb.cols;++i)
			if (intProb.at<double>(0,i)==0)
				intProb.at<double>(0,i)=eps;
		double sum;
		for (int i=0;i!=transProb.rows;++i)
		{
			sum = 0;
			for (int j=0;j!=transProb.cols;++j)
				sum+=transProb.at<double>(i,j);
			for (int j=0;j!=transProb.cols;++j)
				transProb.at<double>(i,j)/=sum;
		}
		for (int i=0;i!=likelihoodProb.rows;++i)
		{
			sum = 0;
			for (int j=0;j!=likelihoodProb.cols;++j)
				sum+=likelihoodProb.at<double>(i,j);
			for (int j=0;j!=likelihoodProb.cols;++j)
				likelihoodProb.at<double>(i,j)/=sum;
		}
		sum = 0;
		for (int j=0;j!=intProb.cols;++j)
			sum+=intProb.at<double>(0,j);
		for (int j=0;j!=intProb.cols;++j)
			intProb.at<double>(0,j)/=sum;
	}

// best state sequence and observation probability using this state sequence -> Viterabi algorithm
double bestStateSequence(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount, unsigned int* bestStates)
{
cv::Mat transProb = A.clone();
		cv::Mat likelihoodProb = B.clone();
		cv::Mat intProb = P.clone();
		model(transProb,likelihoodProb,intProb);
		int seq = observationCount;
		int states = transProb.cols;

		cv::Mat s(states,seq,CV_64F);
        cv::Mat path=Mat::zeros(states,seq,CV_32S);

        cv::Mat newpath=Mat::zeros(states,seq,CV_32S);

		for (int y=0;y!=states;++y)
		{
			s.at<double>(y,0) = intProb.at<double>(0,y) + likelihoodProb.at<double>(y,observations[0]);
			path.at<int>(y,0) = y;
		}
		double maxp,p;
		int state;
		for (int t=1;t!=seq;++t)
		{
			for (int y=0;y!=states;++y)
			{
				maxp = -DBL_MAX;
				state = y;
				for (int y0=0;y0!=states;++y0)
				{
					p = s.at<double>(y0,t-1) + transProb.at<double>(y0,y) + likelihoodProb.at<double>(y,observations[t]);
					if (maxp<p)
					{
						maxp = p;
						state = y0;
					}
				}
				s.at<double>(y,t) = maxp;
				for (int t1=0;t1!=t;++t1)
					newpath.at<int>(y,t1) = path.at<int>(state,t1);
				newpath.at<int>(y,t) = y;
			}
			path.release();
			path = newpath.clone();
		}
		maxp = -DBL_MAX;
		for (int y=0;y!=states;++y)
		{
			if (maxp < s.at<double>(y,seq-1))
			{
				maxp = s.at<double>(y,seq-1);
				state = y;
			}
		}

		Mat statesB = path.row(state).clone();
		for(int i=0;i!=statesB.cols; ++i)
		{
		bestStates[i]=statesB.at<int>(0,i);
		cout<<bestStates[i]<<" ";
	}
	double prob=intProb.at<double>(0,bestStates[0]);
	prob=prob*likelihoodProb.at<double>(bestStates[0],observations[0]);
	for(int i =1 ;i !=observationCount;++i)
	prob=prob*transProb.at<double>(bestStates[i-1],bestStates[i])*likelihoodProb.at<double>(bestStates[i],observations[i]);


	return prob;
}



