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

void generateRandomObservations(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount);
double observationProbabilityForward(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount);
double bestStateSequence(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount, unsigned int* bestStates);
double rd() { return (double)rand() / (double)RAND_MAX; } // we suggest to use this to create uniform random values between 0 and 1

int main(int argc, char** argv)
{
	// keep this to produce our results
	srand(42);
	// Four states, 3 symbols
	Mat A = Mat(4, 4, CV_64F);
	Mat B = Mat(4, 3, CV_64F);
	Mat P = Mat(4, 1, CV_64F);
	A.at<double>(0, 0) = 0.5; //a11
	A.at<double>(0, 1) = 0.2;
	A.at<double>(0, 2) = 0.3;
	A.at<double>(0, 3) = 0.0; //a14

	A.at<double>(1, 0) = 0.2; //a21
	A.at<double>(1, 1) = 0.4;
	A.at<double>(1, 2) = 0.1;
	A.at<double>(1, 3) = 0.3; //a24

	A.at<double>(2, 0) = 0.7; //a31
	A.at<double>(2, 1) = 0.1;
	A.at<double>(2, 2) = 0.1;
	A.at<double>(2, 3) = 0.1;

	A.at<double>(3, 0) = 0.0;
	A.at<double>(3, 1) = 0.1;
	A.at<double>(3, 2) = 0.8;
	A.at<double>(3, 3) = 0.1;

	P.at<double>(0, 0) = 0.7; //Einstieg in die 4 states
	P.at<double>(1, 0) = 0.2;
	P.at<double>(2, 0) = 0.1;
	P.at<double>(3, 0) = 0.0;

	B.at<double>(0, 0) = 0.6; //Output state1
	B.at<double>(0, 1) = 0.2;
	B.at<double>(0, 2) = 0.2;

	B.at<double>(1, 0) = 0.4; //Output state2
	B.at<double>(1, 1) = 0.4;
	B.at<double>(1, 2) = 0.2;

	B.at<double>(2, 0) = 0.3; //Output state3
	B.at<double>(2, 1) = 0.3;
	B.at<double>(2, 2) = 0.4;

	B.at<double>(3, 0) = 0.1; //Output state4
	B.at<double>(3, 1) = 0.2;
	B.at<double>(3, 2) = 0.7;

	// Length = 2;
	unsigned int cnt = 2;
	unsigned int obs1[2]; //Observations sollen 2 sein
	unsigned int bestStates1[2];
	generateRandomObservations(A, B, P, obs1, cnt);
	cout << "Observation Sequence: ";
	for (int i = 0; i < cnt+1; i++)
	{
		cout << obs1[i] << " ";
	}
	cout << endl;
	double prob_all = observationProbabilityForward(A, B, P, obs1, cnt);
	double prob_best = bestStateSequence(A, B, P, obs1, cnt, bestStates1);
	cout << "Probability: " << prob_all << endl;
	cout << "Best Sequence: ";
	for (int i = 0; i < cnt; i++)
	{
		cout << bestStates1[i] << " ";
	}
	cout << "Probability: " << prob_best << endl;
	cout << "Best Prob. / Total Prob: " << prob_best / prob_all << endl << endl;

	// Length = 10
	cnt = 10;
	unsigned int obs2[10];
	unsigned int bestStates2[10];
	generateRandomObservations(A, B, P, obs2, cnt);
	cout << "Observation Sequence: ";
	for (int i = 0; i < cnt; i++)
	{
		cout << obs2[i] << " ";
	}
	cout << endl;
	prob_all = observationProbabilityForward(A, B, P, obs2, cnt);
	prob_best = bestStateSequence(A, B, P, obs2, cnt, bestStates2);
	cout << "Probability: " << prob_all << endl;
	cout << "Best Sequence: ";
	for (int i = 0; i < cnt; i++)
	{
		cout << bestStates2[i] << " ";
	}
	cout << "Probability: " << prob_best << endl;
	cout << "Best Prob. / Total Prob: " << prob_best / prob_all << endl << endl;

	return 0;
}

//Generating random observations by random sampling as learned in the lecture
void generateRandomObservations(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount)
{
	int numStates = A.rows;
	int numSymbols = B.cols;

	int state;

	double r = rd(); //random number
	double s = 0.0; // probability counter
					// Starting state
					// iterate over all states and set the starting state
	for (int i = 0; i < numStates; i++)
	{
		if (r >= s && r < s + P.at<double>(i, 0)) //Einstieg angefangen bei state 1
		{
			state = i;
			break;
		}
		s = s + P.at<double>(i, 0);
	}

	for (int t = 0; t < observationCount; t++) {
		// emit symbol
		r = rd();
		s = 0.0;
		for (int i = 0; i < numSymbols; i++)
		{
			if (r >= s && r < s + B.at<double>(state, i))
			{
				observations[t] = i;
				break;
			}
			s = s + B.at<double>(state, i);
		}
		// switch to next state
		if (t < observationCount - 1)
		{
			r = rd();
			s = 0.0;
			for (int i = 0; i < numStates; i++)
			{
				if (r >= s && r < s + A.at<double>(state, i))
				{
					state = i;
					break;
				}
				s = s + A.at<double>(state, i);
			}
		}
	}
}
// Observation probability -> Forward algorithm
double observationProbabilityForward(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount)
{
	cv::Mat transitionState = A.clone();
	cv::Mat outputProb = B.clone();
	cv::Mat startingProb = P.clone();
	cv::Mat alphaProb = cv::Mat(transitionState.rows, observationCount, CV_64F);

	//Save alpha(1)
	for (int i = 0; i < transitionState.rows; i++)
	{
		//alpha(1) = Phi_i * b_i(o1)
		alphaProb.at<double>(i, 0) = startingProb.at<double>(i, 0)*outputProb.at<double>(i, observations[0]);
	}

	for (unsigned int t = 1; t < observationCount; t++)
	{
		for (int i = 0; i < transitionState.rows; i++)
		{
			alphaProb.at<double>(i, t) = 0;
			for (int j = 0; j < transitionState.rows; j++)
			{
				//alpha(t) = alpha(t-1) * a_all_i * b_i(o in t)
				alphaProb.at<double>(i, t) += alphaProb.at<double>(j, t - 1)*transitionState.at<double>(j, i) * outputProb.at<double>(i, observations[t]);
			}
		}
	}
	double result_prob = 0;
	for (int i = 0; i < transitionState.rows; i++)
	{
		//Sum last alpha(n) per state
		result_prob += alphaProb.at<double>(i, observationCount - 1);
	}

	return result_prob;
}


// best state sequence and observation probability using this state sequence -> Viterbi algorithm
// check https://en.wikipedia.org/wiki/Viterbi_algorithm for a pseudocode example
double bestStateSequence(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount, unsigned int* bestStates)
{
	cv::Mat transitionState = A.clone();
	cv::Mat outputProb = B.clone();
	cv::Mat startingProb = P.clone();
	cv::Mat alphaProb = cv::Mat(transitionState.rows, observationCount, CV_64F);
	cv::Mat path = cv::Mat(transitionState.rows, observationCount, CV_64F);


	//Save alpha(1)
	for (int i = 0; i < transitionState.rows; i++)
	{
		//alpha(1) = Phi_i * b_i(o1)
		alphaProb.at<double>(i, 0) = startingProb.at<double>(i, 0)*outputProb.at<double>(i, observations[0]);
	}

	//First State = 0
	//bestStates[0] = 0;
	for (unsigned int t = 1; t < observationCount; t++)
	{
		double maxT = 0.0;
		for (int i = 0; i < transitionState.rows; i++)
		{
			double tmpTime = 0.0;
			double maxState = 0.0;
			for (int j = 0; j < transitionState.rows; j++)
			{
				//alpha(t) = max (alpha(t-1) * a_all_i * b_i(o in t))
				double tmpState = alphaProb.at<double>(j, t - 1)*transitionState.at<double>(j, i) * outputProb.at<double>(i, observations[t]);

				if (tmpState > maxState)
				{
					//Save index of state t before
					path.at<double>(i, t) = j;
					//Save max Value
					alphaProb.at<double>(i, t) = tmpState;
					maxState = tmpState;
				}
			}
			if (maxState > maxT)
			{
				maxT = maxState;

				//bestStates[t] = i;
			}
		}
	}

	int lastIndex = 0;
	double last_max = 0;
	for (int i = 0; i < transitionState.rows; i++)
	{
		//Sum last alpha(n) per state
		double tmp = alphaProb.at<double>(i, observationCount - 1);
		if (tmp > last_max)
		{
			last_max = tmp;
			bestStates[observationCount - 1] = i;
		}
	}
	double probability = 0;
	for (int t = observationCount-1; t > 0; t--)
	{
		bestStates[t-1] = path.at<double>(bestStates[t], t);
		probability *= alphaProb.at<double>(bestStates[t-1], t);
	}
	return last_max;
}
