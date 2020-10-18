//includes
#include <stdio.h> 																					//Standard I/O library
#include <numeric>																					//For std::accumulate function
#include <string> 																					//For std::to_string function
#include <opencv2/opencv.hpp>																		//opencv libraries
#include "utils.hpp"
#include "ShowManyImages.hpp"
#include <opencv2/videoio.hpp>
#include "FusedTracker.hpp"																		//Header file for GradientTracker

//namespaces
using namespace cv;
using namespace std;
using namespace tracking;

//main function
int main(int argc, char ** argv)
{
	//Path for dataset and result folder
	std::string dataset_path = "/home/smriti/Documents/AVSA/Lab4/AVSA_lab4_datasets/datasets/";		//dataset location.
	std::string output_path = "/home/smriti/Documents/AVSA/Lab4/A/output/";							//location to save output videos

	// Dataset paths
	std::string sequences[] = {"bolt1","road","ball","bag"};										//test data
	std::string image_path = "%08d.jpg"; 															//format of frames. DO NOT CHANGE
	std::string groundtruth_file = "groundtruth.txt"; 												//file for ground truth data. DO NOT CHANGE
	int NumSeq = sizeof(sequences)/sizeof(sequences[0]);											//number of sequences

	/***Loop for all sequence of each category***/
	for (int s=0; s<NumSeq; s++ )
	{
		Mat frame;																					//current Frame
		int frame_idx=0;																			//index of current Frame
		std::vector<Rect> list_bbox_est, list_bbox_gt;												//estimated & ground truth bounding boxes
		std::vector<double> procTimes;																//vector to accumulate processing times

		std::string inputvideo = dataset_path + "/" + sequences[s] + "/img/" + image_path;		 	//path of video. DO NOT CHANGE
		VideoCapture cap(inputvideo);																//reader to grab frames from video

		//check if video exists
		if (!cap.isOpened())
			throw std::runtime_error("Could not open video file " + inputvideo); 					//error if not possible to read video

		// Define the codec and create VideoWriter object
		//The output is stored in 'outcpp.avi' file.
		cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT));	//cv::Size frame_size(700,460);
		VideoWriter outputvideo(output_path+"outvid_" + sequences[s]+".avi",						//xvid compression (cannot be changed in OpenCV)
							VideoWriter::fourcc('X','V','I','D'),10, frame_size);

		//Read ground truth file and store bounding boxes
		std::string inputGroundtruth = dataset_path + "/" + sequences[s] + "/" + groundtruth_file;	//path of groundtruth file. DO NOT CHANGE
		list_bbox_gt = readGroundTruthFile(inputGroundtruth); 									 	//read groundtruth bounding boxes


		std::cout << "Displaying sequence at " << inputvideo  << std::endl;
		std::cout << "  with groundtruth at " << inputGroundtruth << std::endl;

		/***Parameters for analysis***/
		// Feature- Color Space|Gray:1, Blue:2, Green:3, Red:4, 5: Hue, 6: Saturation,7:Value, Default:Gray
		int feature = 1;
		int stride = 2;																				//Stride
		int bins = 9;																				//Number of bins
		int numCandidates =100;	 																//Number of candidates
		char mode = 'F';																			//To choose model
																									//C:Color|G:Gradient|F:Fusion

		//Initialise the object of FusedTracker class with parameters
		FusedTracker Object(mode,list_bbox_gt[0],feature,stride,bins,numCandidates);
		Rect estimated;

		//Main loop for the sequence
		for (;;) {

			//get frame & check if we achieved the end of the video (e.g. frame.data is empty)
			cap >> frame;
			if (!frame.data)
				break;

			//Time measurement
			double t = (double)getTickCount();
			frame_idx=cap.get(cv::CAP_PROP_POS_FRAMES);												//Get the current frame

			/***********Tracking ************/
			estimated = Object.start(frame);											//Finds the bbox final candidate
			list_bbox_est.push_back(estimated);														//Adds bbox to the estimated bbox list
			/********************************/

			//Time measurement
			procTimes.push_back(((double)getTickCount() - t)*1000. / cv::getTickFrequency());
			//std::cout << "  processing time=" << procTimes[procTimes.size()-1] << " ms" << std::endl;

			/*********Plotting****************/

			// plot frame number & groundtruth bounding box for each frame
			putText(frame, std::to_string(frame_idx), cv::Point(10,15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red
			rectangle(frame, list_bbox_gt[frame_idx-1], Scalar(0, 255, 0));		//draw bounding box for groundtruth
			rectangle(frame, list_bbox_est[frame_idx-1], Scalar(0, 0, 255));	//draw bounding box (estimation)
			imshow("Tracking for "+sequences[s]+" (Green=GT, Red=Estimation)", frame);
			outputvideo.write(frame);//save frame to output video
			//exit if ESC key is pressed
			//waitKey();
			if(waitKey(30) == 27) break;

		}

		//comparison groundtruth & estimation
		vector<float> trackPerf = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);

		//Connects the values of similarity map with a line
		int hist_w = 512; int hist_h = 400;									//Width and height of the plot
		int bin_w = cvRound( (double) hist_w/trackPerf.size() );						//Width assigned to each value
		Mat similarity(hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
		for(unsigned int i = 1; i < trackPerf.size(); i++ )
		{
			line(similarity, Point(bin_w*(i-1),hist_h-trackPerf[i-1]*similarity.rows), Point(bin_w*(i),hist_h - trackPerf[i]*similarity.rows), Scalar(0,0,255),2,8,0);
		}
		putText(similarity,"Similarity plot", cv::Point(10,20),FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2,8); //text in white
		imshow( "plot", similarity );
		waitKey();

		//print stats about processing time and tracking performance
		std::cout << "  Average processing time = " << std::accumulate( procTimes.begin(), procTimes.end(), 0.0) / procTimes.size() << " ms/frame" << std::endl;
		std::cout << "  Average tracking performance = " << std::accumulate( trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size() << std::endl;

		//release all resources
		cap.release();			// close inputvideo
		outputvideo.release(); 	// close outputvideo
		destroyAllWindows(); 	// close all the windows
	}
	printf("Finished program.");
	return 0;
}
