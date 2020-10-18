#include <opencv2/opencv.hpp>
#include "GradientTracker.hpp"

using namespace gradtrack;
using namespace std;
using namespace cv;

GradientTracker::GradientTracker(Rect bbox, int stride, int bins, int numCandidates)
{
	model = bbox;								//Bbox of model
	last_obs = bbox;							//Bbox of final candidate
	this->stride = stride;						//Stride for scanning the neighbourhood
	this->bins = bins;							//Histogram Bins
	this->numCandidates = numCandidates;		//Number of candidate regions
	calc_bounds_candidates();					//Calculate the neighbourgood region

	//Initialising descriptor of Histogram of Gradien
	descriptor = HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), bins, 1);
}
//To calculate the neighbourhood
void GradientTracker::calc_bounds_candidates() {
	int N = sqrt(numCandidates);
	indx_start = (N / 2) * stride;
	if (N % 2 == 0) {
		// If even grid then left to right search wont be the same		
		indx_last = ((N / 2) - 1) * stride;
	}
	else {
		indx_last = (N / 2) * stride;
	}
}

//To get center of a bounding box
Point GradientTracker::get_center(Rect box) {
	float x = box.x + box.width * 0.5;
	float y = box.y + box.height * 0.5;
	return Point(x, y);
}

//To get rectangle coordinates from center, width and height
Rect GradientTracker::get_rect(Point center, int width, int height) {
	float xmin = center.x - width * 0.5;
	float ymin = center.y - height * 0.5;
	return Rect(xmin, ymin, width, height);
}

//To generate list of candidate regions
void GradientTracker::gen_candidates() {

	// Assumption that we only check NxN grid neighbours
	// NumCandidates=n, Grid = sqrt(n)xsqrt(n). (including the center)
	Point center = get_center(last_obs);
	for (int i = center.x - indx_start; i <= center.x + indx_last; i += stride) {
		for (int j = center.y - indx_start; j <= center.y + indx_last; j += stride) {
			Rect rect = get_rect(Point(i, j), last_obs.width, last_obs.height);
			if(rect.x>0 && rect.y>0 && rect.x+last_obs.width<current_frame.cols&& rect.y+last_obs.height<current_frame.rows)
				current_neighbours.push_back(rect);
		}
	}
}

//To compute the HOG descriptor
vector<float> GradientTracker::get_gradient(Rect bbox) {
	Mat cropImg;
	current_frame(bbox).copyTo(cropImg);
	resize(cropImg, cropImg, Size(64, 128));
	vector<float> descs;	
	
	descriptor.compute(cropImg, descs, Size(8, 8), Size(0, 0));
	return descs;
}

// Bhattacharya Distance between the given two normalized histograms
double GradientTracker::calc_distance(vector<float> hist1, vector<float> hist2) {
	return norm(hist1, hist2, NORM_L2);
}

//Process Function for Gradient Based Tracker
Rect GradientTracker::start(Mat current_frame) {

	cvtColor(current_frame, this->current_frame, cv::COLOR_RGB2GRAY);
	if (model_hist==vector<float>()) {
		model_hist = get_gradient(model);
	}

	//Generate candidates. Empty the previous list
	current_neighbours.clear();
	gen_candidates();

	candidate_scores.clear();
	for (unsigned int i = 0; i < current_neighbours.size(); i++) {
		candidate_scores.push_back(calc_distance(model_hist, get_gradient(current_neighbours[i])));
	}

	int minElem = min_element(candidate_scores.begin(), candidate_scores.end()) - candidate_scores.begin();
	last_obs = current_neighbours[minElem];
	return last_obs;
}

//Returns the candidate regions in the neighbourhood
vector<Rect> GradientTracker::get_neighbours() {
	return current_neighbours;
}

//Get normalised L1 scores
vector<float> GradientTracker::get_norm_scores() {
	normalize(candidate_scores, candidate_scores, 1.0, 0.0, NORM_L1);
	return candidate_scores;
}

