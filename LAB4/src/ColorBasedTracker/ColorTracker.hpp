#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace tracking {
	class ColorTracker {
	private:
		int feature;					//Integer representing the choice of color based feature
		int stride;						//Stride
		int bins;						//Histogram bins
		int numCandidates;				//Number of candidates
		int indx_start, indx_last;		//Index of neighborhood
		float range;					//Range of values to create histogram

		Rect model;						//Model Candidate bounding box
		Rect last_obs;					//Estimated Candidate bounding box
		Rect current_box;				//Current Candidate bounding box
		vector<Rect> current_neighbours;//Candidate Regions
		Mat current_frame;				//Current frame
		Mat model_hist;					//Histogram of the model
		vector<float> candidate_scores;	//Battacharya distance corresponding to models

	public:
		ColorTracker(Rect bbox, int feature=5, int stride=2, int bins=16, int numCandidates=121);

		// Helper functions
		void gen_candidates();
		Point get_center(Rect box);
		Rect get_rect(Point center, int width, int height);
		Mat get_feature(Rect bbox);
		Mat get_histogram(Rect bbox);
		float calc_distance(Mat hist1, Mat hist2);
		void calc_bounds_candidates();
		vector<Rect> get_neighbours();
		vector<float> get_scores();

		//Process functions
		Rect start(Mat current_frame, bool& first_frame);
	};
}
