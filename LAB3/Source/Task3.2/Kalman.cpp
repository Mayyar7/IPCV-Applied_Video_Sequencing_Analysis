#include "kalman.hpp"
#include "ShowManyImages.hpp"
using namespace cv;
using namespace std;

/*Sets the values of private members, calls the function to initialise the parameters of Kalman Filter, calls the function to initialise the position of object*/
KalmanFilterOp::KalmanFilterOp(int stateSize, int measSize, unsigned int type,bool vel_flag)
{
	KalmanFilterOp::stateSize = stateSize;
	KalmanFilterOp::measSize = measSize;
	State = Mat::zeros(stateSize, 1, type);
	Meas = Mat::zeros(measSize, 1, type);
	InitialiseKalman(vel_flag, type);
}

/*Initializes parameters of Kalman filter*/
void KalmanFilterOp::InitialiseKalman(bool vel_flag,unsigned int type)
 {

		int csize = 0;

		//Initialization
		if (vel_flag) {

			kf.init(stateSize, measSize, csize, type);
			//H
			//kf.measurementMatrix = Mat::zeros(measSize, stateSize, type);
			kf.measurementMatrix.at<float>(0) = 1.0f;
			kf.measurementMatrix.at<float>(6) = 1.0f;

			//A
			kf.transitionMatrix = Mat::zeros(stateSize, stateSize, type);
			cv::setIdentity(kf.transitionMatrix, Scalar::all(1.0f));
			kf.transitionMatrix.at<float>(1) = 1.0f;
			kf.transitionMatrix.at<float>(11) = 1.0f;



			//Q
			kf.processNoiseCov = Mat::zeros(stateSize, stateSize, type);
			kf.processNoiseCov.at<float>(0) = 25.0f;
			kf.processNoiseCov.at<float>(5) = 10.0f;
			kf.processNoiseCov.at<float>(10) = 25.0f;
			kf.processNoiseCov.at<float>(15) = 10.0f;

			//P
			kf.errorCovPost = Mat::zeros(stateSize, stateSize, type);
			cv::setIdentity(kf.errorCovPost, Scalar::all(1e5f));
			}

			else {
			kf.init(stateSize, measSize, csize, type);

			//H
			kf.measurementMatrix = Mat::zeros(measSize, stateSize, type);
			kf.measurementMatrix.at<float>(0) = 1.0f;
			kf.measurementMatrix.at<float>(9) = 1.0f;

			//A
			kf.transitionMatrix = Mat::zeros(stateSize, stateSize, type);
			cv::setIdentity(kf.transitionMatrix, Scalar::all(1.0f));
			kf.transitionMatrix.at<float>(1) = 1.0f;
			kf.transitionMatrix.at<float>(2) = 0.5f;
			kf.transitionMatrix.at<float>(8) = 1.0f;
			kf.transitionMatrix.at<float>(22) = 1.0f;
			kf.transitionMatrix.at<float>(23) = 0.5f;
			kf.transitionMatrix.at<float>(29) = 1.0f;

			//Q
			kf.processNoiseCov = Mat::zeros(stateSize, stateSize, type);
			kf.processNoiseCov.at<float>(0) = 25.0f;
			kf.processNoiseCov.at<float>(7) = 10.0f;
			kf.processNoiseCov.at<float>(14) = 1.0f;
			kf.processNoiseCov.at<float>(21) = 25.0f;
			kf.processNoiseCov.at<float>(28) = 10.0f;
			kf.processNoiseCov.at<float>(35) = 1.0f;


			//P
			kf.errorCovPost = Mat::zeros(stateSize, stateSize, type);
			cv::setIdentity(kf.errorCovPost, Scalar::all(1e5f));

		}

		//R
		kf.measurementNoiseCov = Mat::zeros(2, 2, type);

		/* R to set for each video in Task 3.2: Video2-[25, 0;0,25], Video3,Video5 - [250,0;0,250], Video6- [500,0;0, 500]*/
		cv::setIdentity(kf.measurementNoiseCov, Scalar::all(25.0f));
	}
//Predict the state
void KalmanFilterOp::ApplyKalmanPrediction()
 {
	State = kf.predict();
 }

//Correct the state
void KalmanFilterOp::ApplyKalmanCorrection(std::vector<Point> &centerList, std::vector<Point> &predictedList,std::vector<Point> &correctedList, cvBlob Blob)
{
	bool pred = true;				 													//True: When measurement does not exist; False: When measurement exists and correction stage has to be implemented

	if(Blob.w * Blob.h !=0)																//To check if blob is found, area_of_blob >0
	{
		pred = false;
	}
	int index = (stateSize -   measSize) / 2 + 1;
	predictedList.push_back(Point(State.at<float>(0), State.at<float>(index)));

	if(!pred)
	{
		State = kf.correct(Meas);													//Correction step
		correctedList.push_back(Point(State.at<float>(0), State.at<float>(index)));
		centerList.push_back(Point(State.at<float>(0), State.at<float>(index)));
	}
	else
	{
		centerList.push_back(Point(State.at<float>(0), State.at<float>(index)));
	}
}

/*Setter function for measurement: Measurement is the center of blob detected*/
void KalmanFilterOp::setMeas(cvBlob blob,std::vector<Point>  &measureList )
{
	Meas.at<float>(0) = (float)(blob.x + blob.w/2);
	Meas.at<float>(1) = (float)(blob.y + blob.h/2);
	measureList.push_back(Point(Meas.at<float>(0), Meas.at<float>(1)));
}

/*Initial position of the object: Set as the center of first bounding box*/
void KalmanFilterOp::setInitialState()
{
	int index = (stateSize -   measSize) / 2 + 1;
	State.at<float>(0) = Meas.at<float>(0);
	State.at<float>(index) = Meas.at<float>(1);
	kf.statePost = State;
}

/*Function implementing all stages of Kalman Filter and showing the results*/

void KalmanFilterOp::Start(cv::Mat frame,std::vector<Point>  &centerList, std::vector<Point> &predictedList, std::vector<Point>  &correctedList,std::vector<Point>  &measureList, cvBlob blob, int it)
{

		setMeas(blob, measureList);
		if(flag)
		{
			setInitialState();
			flag = false;
		}
		ApplyKalmanPrediction();
		ApplyKalmanCorrection(centerList, predictedList, correctedList,blob);

		/***To display images with text***/
		Mat detected_blob= paintBlobImage(frame, blob); //Detected Blobs
		cv::putText(detected_blob, "Detected Blob", Point2f(20, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 255), 3);

		Mat trajectory= paintFinalTrajectory(frame, centerList);
		cv::putText(trajectory, "Trajectory detected", Point2f(20, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 0, 0), 3);

		Mat intermediate = paintCirclesImage(frame, centerList, measureList);
		char str1[200], str2[200];

		snprintf(str1,200, "Measurement: (%d),(%d)", measureList.back().x, measureList.back().y);
		snprintf(str2, 200, "Estimated: (%d),(%d)", centerList.back().x, centerList.back().y);
		cv::putText(intermediate, str1, Point2f(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255, 0), 3);
		cv::putText(intermediate, str2, Point2f(20, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 3);
		char str[200];
		snprintf(str,200,"Frame No:%d", it);
		cv::putText(frame, str, Point2f(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3);

		ShowManyImages("Results", 4, frame,detected_blob , intermediate, trajectory);


}
