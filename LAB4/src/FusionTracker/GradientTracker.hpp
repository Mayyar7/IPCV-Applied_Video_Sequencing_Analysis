#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace gradtrack {
	class GradientTracker {
	private:
		int stride;						//Stride
		int bins;						//Histogram bins
		int numCandidates;				//Number of candidates
		int indx_start, indx_last;		//Index of neighborhood

		Rect model;						//Model Candidate bounding box
		Rect last_obs;					//Estimated Candidate bounding box
		vector<Rect> current_neighbours;//Candidate Regions
		Mat current_frame;				//Current frame
		vector<float> model_hist;		//To store HOG features of model
		HOGDescriptor descriptor;		//Object
		vector<float> candidate_scores;	//Battacharya distance corresponding to models

		public:
		GradientTracker(Rect bbox = Rect(), int stride=2, int bins=9, int numCandidates=100);

		// Helper functions
		void gen_candidates();
		Point get_center(Rect box);
		Rect get_rect(Point center, int width, int height);
		vector<float> get_gradient(Rect bbox);
		double calc_distance(vector<float> hist1, vector<float> hist2);
		void calc_bounds_candidates();

		vector<Rect> get_neighbours();
		vector<float> get_norm_scores();

		//Process function
		Rect start(Mat current_frame);
	};
}
