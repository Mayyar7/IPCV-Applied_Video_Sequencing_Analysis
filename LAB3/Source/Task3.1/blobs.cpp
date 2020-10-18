#include "blobs.hpp"
using namespace cv;
using namespace std;

 Mat paintBlobImage(cv::Mat frame, cvBlob &blob)
{
	cv::Mat blobImage;
	frame.copyTo(blobImage);
	Scalar color;
	std::string label="";
	color = Scalar(255, 255, 255);

	Point p1 = Point(blob.x, blob.y);
	Point p2 = Point(blob.x+blob.w, blob.y+blob.h);

	rectangle(blobImage, p1, p2, color, 1, 8, 0);
	rectangle(blobImage, p1, p2, Scalar(255, 255, 255), 1, 8, 0);

	return blobImage;
}

 /***function to show predicted, corrected and final centers on the current frame***/

 Mat paintCirclesImage(cv::Mat frame,std::vector<Point> centerList,std::vector<Point> measuredList)
 {
	 cv::Mat blobImage;
	 frame.copyTo(blobImage);
	 for(unsigned int i = 0; i <measuredList.size(); i++)
	 {
		 Point pt = measuredList[i];
		 circle(blobImage,pt, 14,Scalar(0, 255,0), 2, 8, 0 );					//measured center in color green

	 }

	 for(unsigned int i = 0; i < centerList.size(); i++)
	 {
		 Point pt = centerList[i];
		 circle(blobImage,pt, 8,Scalar(255, 0, 0), -1, 8, 0 );					//predicted center after correction with measurementin color blue

	 }
	 return blobImage;
 }

 /***function to show final trajectory on the current frame***/
 Mat paintFinalTrajectory(cv::Mat frame,std::vector<Point> centerList)
 {
	 cv::Mat blobImage;
	 frame.copyTo(blobImage);
	 for(unsigned int i = 0; i < centerList.size()-1; i++)
	 {
		 line(blobImage,centerList[i],centerList[i+1],Scalar(255, 0, 0), 4); 	//final trajectory in color blue

	 }
	 return blobImage;
 }
 /***Function to extract blobs from foreground segmentation***/

 void extractBlobs(cv::Mat fgmask, cvBlob &blob, int connectivity, int min_width, int min_height)
 {
	Mat aux= Mat::zeros(Size(fgmask.cols, fgmask.rows), CV_32SC1); 				// image to be updated each time a blob is detected (blob cleared)
	fgmask.convertTo(aux,CV_32SC1);

	Mat mask;
    mask.create(fgmask.rows + 2, fgmask.cols + 2, CV_8UC1);

	aux = (aux / 255); 															// Converts the frame to have only 0 and 1. 0 - background and shadow, 1 - foreground
	int ID = 0; 																// Variable for painting connected components
	Rect r; 																	// bounding rectangle for the object detected
	Mat Points; 																//Variable to find the painted area

	/**Connected component analysis**/

	blob = initBlob(0,0,0,0,0);
	 for (int i = 0; i < aux.rows; i++)
	 {
		 for (int j = 0; j < aux.cols; j++)
		 {

			 if (aux.at<int>(i, j) == 1)
			 {
				 mask = Scalar::all(0);
				 ID += 1;
				 floodFill(aux, mask, Point(j, i),Scalar(1), &r, Scalar(), Scalar(), connectivity| cv::FLOODFILL_MASK_ONLY);

				 if (r.area()> blob.w*blob.h && r.height> min_height && r.width > min_width )
				{
					blob = initBlob(ID, r.x, r.y, r.width, r.height);			// add to list: (i)if the blob has min height and width (ii) if the blob is larger than the previous blob

				}
				 Range rows(1, mask.rows - 1);
				 Range columns(1, mask.cols - 1);
				 Mat new_mask = mask(rows, columns);
				 new_mask.convertTo(new_mask, CV_32SC1);
				 aux = aux.mul(1-new_mask);
           }
		}

	}
 }






