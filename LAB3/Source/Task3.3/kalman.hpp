//Kalman Filter Operation Class
#include "opencv2/opencv.hpp"
using namespace cv;
#include "blobs.hpp"

class KalmanFilterOp
{
private:
	int stateSize;				//Stores the size of state variable
	int measSize;				//Stores the size of measurement variable
	Mat State;					//Stores the current state
	Mat Meas;					//Stores the current measurement
	KalmanFilter kf;			//Kalman Filter object
	bool flag = true;
public:
	KalmanFilterOp(int stateSize, int measSize, unsigned int type,bool vel_flag);																									//Constructor
	void InitialiseKalman(bool vel_flag,unsigned int type);																															//To initialise value of Kalman Filter parameters
	void ApplyKalmanPrediction();																																					//Prediction Stage
	void ApplyKalmanCorrection(std::vector<Point> &centerList,std::vector<Point> &predictedList,std::vector<Point> &correctedList,cvBlob Blob);										//Correction Stage
	void setInitialState();																																							//Initial position of the object
	void setMeas(cvBlob blob, std::vector<Point>  &measureList);																												    //Setter for measurement
	void Start(cv::Mat frame,std::vector<Point>  &centerList, std::vector<Point> &predictedList, std::vector<Point>  &correctedList,												//To implement functions for Kalman Filter
			std::vector<Point>  &measureList, cvBlob blob, int it);
};
