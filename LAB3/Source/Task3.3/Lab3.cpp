
//system libraries C/C++
#include <stdio.h>
#include <iostream>
#include <sstream>

//opencv libraries
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

//Header Files
#include "ShowManyImages.hpp"
#include "blobs.hpp"
#include "kalman.hpp"

//Namespaces
using namespace cv;
using namespace std;

//main function
int main(int argc, char ** argv)
{
	//Variables for execution time

	double t, acum_t;
	int t_freq = getTickFrequency();
	int it=1;
	acum_t=0;

	//Reading Video Sequence

	VideoCapture cap;															//reader to grab video frames
	string inputvideo = "/home/smriti/Documents/AVSA/Lab3/AVSA_Lab3_datasets/dataset_lab3/lab3.3/abandonedBox_600_1000_clip.mp4";
	if(argc==2)
	{
		inputvideo = argv[1];
	}
	cout << "Accessing sequence at " << inputvideo << endl;
	cap.open(inputvideo);
	if (!cap.isOpened()) {
		cout << "Could not open video file " << inputvideo << endl;
		return -1;
	}


	Mat frame; 																	// current Frame
	Mat fgmask; 																// foreground mask
	cvBlob blob;																// blob
	std::vector<Point> centerList,predictedList, correctedList, measuredList;					// centerList: Finally obtained centers, predList: Predict centers, CorrectedList: Correction of predicted centers with measurements

	/****Variables for foreground segmentation: MOG2 approach****/

	Ptr<BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2();
	int connectivity = 8;														// 4 or 8

	/* Learning rate to set for each video in Task Task 3.3: abandonedBox: -1, boats: 0.0001, pedestrians:0.001,street_corner:0.001*/
	double learningrate = -1; 												//default value (as starting point)
	/*history to set for each video in Task 3.3: abandonedBox: default, boats: 100, pedestrians:default,street_corner:100(for acceleration)*/
	double history = 50;
	double varThreshold = 16;

	/*min_width, min_height to set for each video in Task 3.3: abandonedBox: 15, boats: 30, pedestrians,street_corner:25*/

	int min_width = 15; 														// the minimum width for the blob to be added to the list
	int min_height = 15; 														// the minimum height for the blob to be added to the list
	pMOG2.dynamicCast<cv::BackgroundSubtractorMOG2>()->setHistory(history);
	pMOG2.dynamicCast<cv::BackgroundSubtractorMOG2>()->setVarThreshold(varThreshold);
	Mat element;
	/**For pedestrians in task3.3, set kernel size as (7,7)***/
	element = getStructuringElement(MORPH_RECT, Size(3, 3));

	/****Variables for object Ttacking: Kalman Filter****/

	bool vel_flag = false; 														//Flag to switch between models. True: Constant Velocity Model, False: Constant Acceleration Model
	int measSize;
	int stateSize ;
	unsigned int type = CV_32F;

	if(vel_flag){measSize =2;stateSize=4 ;}
	else{measSize =2;stateSize=6 ;}

	KalmanFilterOp Obj(stateSize, measSize, type,vel_flag);						//Object for KalmanFilterOp Class

	/****Main loop****/
	bool flag= false;
	Mat img; 																	//Current Frame
	while(true)
	{
			cap >> img;															//Get frame
			if (!img.data)														//Check if we achieved the end of the file (e.g. img.data is empty)
				break;

			t = (double)getTickCount();											//Time measurement

			/****Foreground segmentation****/
			img.copyTo(frame);
			pMOG2->apply(frame, fgmask, learningrate);
			morphologyEx(fgmask, fgmask, MORPH_OPEN, element);

			extractBlobs(fgmask, blob, connectivity,min_width,min_height);

			/****Object Tracking****/
			if(blob.w*blob.h>0)
			{
				flag = true;

			}
			if(flag)
			{
				Obj.Start(frame,centerList,predictedList,correctedList,measuredList,blob, it);
			}

			/****Time measurement****/
			t = (double)getTickCount() - t;
			acum_t=+t;
			it++;
			if(waitKey(30) == 27) break;
		}

	cout << it-1 << "frames processed in " << 1000*acum_t/t_freq << " milliseconds."<< endl;


	//release all resources
	waitKey(0);
	cap.release();
	destroyAllWindows();

return 0;
}

