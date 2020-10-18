#include <opencv2/opencv.hpp>
#include "FusedTracker.hpp"
#include "ColorTracker.hpp"
#include "GradientTracker.hpp"
using namespace std;
using namespace cv;
using namespace tracking;
using namespace colortrack;
using namespace gradtrack;

//Constructor that initialises the parameters for the tracker
FusedTracker::FusedTracker(char mode , Rect bbox , int feature, int stride, int bins, int numCandidates) {
	this->mode = mode;
	colorTrack = ColorTracker(bbox, feature, stride, bins, numCandidates);
	gradTrack = GradientTracker(bbox, stride, bins, numCandidates);

}

//Implements tracker based on mode parameter
Rect FusedTracker::start(Mat frame) {

	frame.copyTo(this->current_frame);

	switch(mode)
	{
	case ('C'): {
		// Only colour
		last_obs = colorTrack.start(current_frame);
		break;
	}
	case ('G'): {
		// Only gradient
		last_obs = gradTrack.start(current_frame);
		break;
	}
	default: {

		last_obs = combine_results();
		break;
	}
	}
	return last_obs;
}

//Function to fuse the color and gradient based tracker
Rect FusedTracker::combine_results() {
	// Calculate with both the trackers
	Rect temp;
	temp = colorTrack.start(current_frame);
	temp = gradTrack.start(current_frame);
	neighbours = colorTrack.get_neighbours();
	vector<float> cscores = colorTrack.get_norm_scores();
	vector<float> gscores = gradTrack.get_norm_scores();

	// Add the two scores
	transform(cscores.begin(), cscores.end(), gscores.begin(), cscores.begin(), std::plus<float>());

	int minElem = min_element(cscores.begin(), cscores.end()) - cscores.begin();
	return neighbours[minElem];
}

