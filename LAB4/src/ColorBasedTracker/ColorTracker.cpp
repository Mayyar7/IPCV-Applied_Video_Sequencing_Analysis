#include <opencv2/opencv.hpp>
#include "ColorTracker.hpp"
#include "ShowManyImages.hpp"

using namespace tracking;
using namespace std;
using namespace cv;

//Constructs initialised member variables with parameters
ColorTracker::ColorTracker(Rect bbox, int feature, int stride, int bins, int numCandidates)
{
	model = bbox;								//Bbox of model
	last_obs = bbox;							//Bbox of final candidate
	this->feature = feature;					//Integer representing color space feature used
	this->stride = stride;						//Stride for scanning the neighbourhood
	this->bins = bins;							//Histogram Bins
	this->numCandidates = numCandidates;		//Number of candidate regions
	calc_bounds_candidates();					//Calculate the neighbourgood region
}

//To calculate the neighbourhood
void ColorTracker::calc_bounds_candidates() {
	int N = sqrt(numCandidates);
	indx_start = (N / 2) * stride;
	if (N % 2 == 0 ) {
		// If even grid then left to right search wont be the same		
		indx_last = ((N / 2) - 1) * stride;
	}
	else {
		indx_last = (N / 2) * stride;
	}
}

//To get center of a bounding box
Point ColorTracker::get_center(Rect box) {
	float x = box.x + box.width * 0.5;
	float y = box.y + box.height * 0.5;
	return Point(x, y);
}

//To get rectangle coordinates from center, width and height
Rect ColorTracker::get_rect(Point center, int width, int height) {
	float xmin = center.x - width * 0.5;
	float ymin = center.y - height * 0.5;
	return Rect(xmin, ymin, width, height);
}

//To generate list of candidate regions
void ColorTracker::gen_candidates() {

	// Assumption that we only check NxN grid neighbours
	// NumCandidates=n, Grid = sqrt(n)xsqrt(n). (including the center)

	Point center = get_center(last_obs);
	for (int i = - indx_start; i <=indx_last; i += stride) {
		for (int j = - indx_start; j <= indx_last; j += stride) {
			current_neighbours.push_back(get_rect(Point(center.x + i, center.y + j), last_obs.width, last_obs.height));
		}
		
	}
}

//Returns the candidate region in appropriate feature space
Mat  ColorTracker::get_feature(Rect bbox){

	Mat cropImg, featImg;
	current_frame(bbox).copyTo(cropImg);
	
	range = 256;												//Default value

	//GrayScale
	if (feature == 1) {
		cvtColor(cropImg, featImg, COLOR_BGR2GRAY);
	}

	//RGB Channel
	else if (feature > 1 && feature <= 4) {

		Mat bgr_channels[3];
		split(cropImg, bgr_channels);

		switch (feature) {
		case(2): {
			bgr_channels[0].copyTo(featImg); 					//Blue Channel
			break;
		}
		case(3): {
			bgr_channels[1].copyTo(featImg); 					//Green Channel
			break;
		}
		case(4): {
			bgr_channels[2].copyTo(featImg); 					//Red Channel
			break;
		}
		default:
			break;

		}
	}
	//HSV Channel
	else if (feature > 4 && feature <= 7) {

		Mat  HSVImg, hsv_channels[3];
		cvtColor(cropImg, HSVImg, COLOR_BGR2HSV);
		split(HSVImg, hsv_channels);
		switch (feature) {
		case(5): {
			range = 180;										//Set range to 180
			hsv_channels[0].copyTo(featImg); 					//Hue Channel
			break;
		}
		case(6): {
			hsv_channels[1].copyTo(featImg);					//Saturation Channel
			break;
		}
		case(7): {
			hsv_channels[2].copyTo(featImg);					//Value Channel
			break;
		}
		default:
			break;
		}
	}
	else {

		cvtColor(cropImg, featImg, COLOR_BGR2GRAY); 			//GrayScale if invalid feature value!

	}

	return featImg;

}

//To generate the histogram of a region
Mat ColorTracker::get_histogram(Rect bbox) {

	//Obtains the candidate image in chosen color space
	Mat featImg = get_feature(bbox);

	Mat hist;
	float range_hist[] = { 0, range }; 									//the upper boundary is exclusive
	const float* histRange = { range_hist };

	calcHist(&featImg, 1, 0, Mat(), hist, 1, &bins, &histRange);		//Histogram computation
	normalize(hist, hist, 1, 0, NORM_L1, -1, Mat());					//Normalisation using L1 norm

	return hist;
}

//Battacharya Distance between the given two normalized histograms
float ColorTracker::calc_distance(Mat hist1, Mat hist2) {
	return compareHist(hist1,hist2,HISTCMP_BHATTACHARYYA);
}

//Process function for color based histogram tracking
Rect ColorTracker::start(Mat current_frame, bool& first_frame) {
	current_frame.copyTo(this->current_frame);
	if (first_frame) {
		first_frame = false;
		model_hist = get_histogram(model);
	}
	
	//Generate candidates. Empty the previous list
	current_neighbours.clear();
	gen_candidates();
	candidate_scores.clear();


	//Loop over candidates
	for (unsigned int i = 0; i < current_neighbours.size(); i++) {
		candidate_scores.push_back(calc_distance(model_hist,
					get_histogram(current_neighbours[i]))); 		//Store the battacharya distance in a list

	}

	//Calculates the minimum battacharya distance from the list
	int minElem = min_element(candidate_scores.begin(), candidate_scores.end()) - candidate_scores.begin();
	//Selects the candidate corresponding to the minimum distance
	last_obs = current_neighbours[minElem];



	/****** Plotting********/

	//Code for plotting the histogram
	int hist_w = 512; int hist_h = 400;									//Width and height of the plot
	int bin_w = cvRound( (double) hist_w/bins );						//Width of each bin

	Mat histImage_mod(hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );		//Template to plot model histogram
	Mat histImage_can(hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );		//Template to plot candidate histogram

	Mat fin_hist_can = get_histogram(last_obs);							//Histogram for selected candidate
	fin_hist_can = histImage_can.rows*fin_hist_can;						//To plot the normalized histogram


	//Connects the values of histogram bin with a line
	for( int i = 1; i < bins; i++ )
	{
	    line( histImage_mod, Point( bin_w*(i-1), hist_h - cvRound(model_hist.at<float>(i-1)*histImage_mod.rows) ) ,
							 Point( bin_w*(i), hist_h - cvRound(model_hist.at<float>(i)*histImage_mod.rows) ),
							 Scalar( 0, 255, 0), 2, 8, 0  );
	    line( histImage_can, Point( bin_w*(i-1), hist_h - cvRound(fin_hist_can.at<float>(i-1))) ,
	                     Point( bin_w*(i), hist_h - cvRound(fin_hist_can.at<float>(i)) ),
	                     Scalar( 0,255, 0), 2, 8, 0  );
	}

	//Final Candidate
	Mat fin_cand;
	current_frame(last_obs).copyTo(fin_cand);

	//Feature
	Mat feat_cand;
	get_feature(last_obs).copyTo(feat_cand);

	//Plotting the candidate bounding box on the current frame
	rectangle(current_frame, last_obs, Scalar(0, 0, 255));

	/****Plot using ShowManyImages****/
	putText(histImage_can,"Candidate Histogram", cv::Point(10,25),FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255),3,8); //text in white
	putText(histImage_mod,"Model Histogram", cv::Point(10,25),FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255),3,8); 	 //text in white


	//Current frame, Selected Candidate, Selected Candidate in Feature Space, Histogram
	ShowManyImages("Visualisation", 5, current_frame,fin_cand, feat_cand, histImage_can,histImage_mod);

	return last_obs;
}

//Returns the candidate regions in the neighbourhood
vector <Rect> ColorTracker::get_neighbours() {
		return current_neighbours;
}

//Returns the Battacharya scores
vector<float> ColorTracker::get_scores() {
	normalize(candidate_scores, candidate_scores, 1.0, 0.0, NORM_L1);
	return candidate_scores;
}
