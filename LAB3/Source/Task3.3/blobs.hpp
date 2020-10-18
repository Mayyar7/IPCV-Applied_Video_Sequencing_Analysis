#ifndef BLOBS_H_INCLUDE
#define BLOBS_H_INCLUDE

#include "opencv2/opencv.hpp"
using namespace cv;
const int MAX_FORMAT = 1024;

/// Type of labels for blobs

struct cvBlob {
	int     ID;  						//blob ID
	int   x, y;  						//blob position
	int   w, h;  						//blob sizes
	char format[MAX_FORMAT];
};

inline cvBlob initBlob(int id, int x, int y, int w, int h)
{
	cvBlob B = { id,x,y,w,h};
	return B;
}

/*
* Headers of blob-based functions
*
*/

//blob drawing functions
Mat paintBlobImage(Mat frame,  cvBlob &blob);
Mat paintCirclesImage(cv::Mat frame,std::vector<Point> centerList,std::vector<Point> measuredList);
Mat paintFinalTrajectory(cv::Mat frame,std::vector<Point> centerList);

//blob extraction functions
void extractBlobs(cv::Mat fgmask, cvBlob &blob, int connectivity, int min_width, int min_height);
#endif
