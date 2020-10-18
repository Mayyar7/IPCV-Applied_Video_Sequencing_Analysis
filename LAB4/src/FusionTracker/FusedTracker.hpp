#include <opencv2/opencv.hpp>
#include "ColorTracker.hpp"
#include "GradientTracker.hpp"
using namespace std;
using namespace cv;
using namespace colortrack;
using namespace gradtrack;

namespace tracking {
	class FusedTracker {

	private:
		char mode;							//To choose whether color based, gradient based or fusion
		vector <float> combined_scores;		//To store the combined scores from Gradient and Color
		bool first_frame;					//first_frame of the sequence
		Mat current_frame;					//Stores current frame
		Rect last_obs;						//Final candidate selected
		vector<Rect> neighbours;			//Candidate Regions
		ColorTracker colorTrack;			//Color Tracker Object
		GradientTracker gradTrack;			//Gradient Tracker Object


	public:
		//Constructor
		FusedTracker(char mode = 'C',Rect bbox=Rect(), int feature = 5, int stride = 2, int bins = 16, int numCandidates = 121);
		//Process function
		Rect start(Mat frame);
		//Helper Function
		Rect combine_results();
	};
}
