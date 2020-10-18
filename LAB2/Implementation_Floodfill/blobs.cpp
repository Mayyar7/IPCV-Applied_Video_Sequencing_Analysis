/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB2: Blob detection & classification
 *	Lab2.0: Sample Opencv project
 *
 *
 * Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es), Juan C. San Miguel (juancarlos.sanmiguel@uam.es)
 */

#include "blobs.hpp"
using namespace cv;
using namespace std;
/**
 *	Draws blobs with different rectangles on the image 'frame'. All the input arguments must be
 *  initialized when using this function.
 *
 * \param frame Input image
 * \param pBlobList List to store the blobs found
 * \param labelled - true write label and color bb, false does not wirite label nor color bb
 *
 * \return Image containing the draw blobs. If no blobs have to be painted
 *  or arguments are wrong, the function returns a copy of the original "frame".
 *
 */
 Mat paintBlobImage(cv::Mat frame, std::vector<cvBlob> bloblist, bool labelled)
{
	cv::Mat blobImage;
	//check input conditions and return original if any is not satisfied
	//...
	frame.copyTo(blobImage);
	//required variables to paint
	//...

	//paint each blob of the list
	for(int i = 0; i < bloblist.size(); i++)
	{
		cvBlob blob = bloblist[i]; //get ith blob
		//...
		Scalar color;
		std::string label="";
		switch(blob.label){
		case PERSON:
			color = Scalar(255,0,0);
			label="PERSON";
			break;
		case CAR:
					color = Scalar(0,255,0);
					label="CAR";
					break;
		case OBJECT:
					color = Scalar(0,0,255);
					label="OBJECT";
					break;
		default:
			color = Scalar(255, 255, 255);
			label="UNKOWN";
		}

		Point p1 = Point(blob.x, blob.y);
		Point p2 = Point(blob.x+blob.w, blob.y+blob.h);

		rectangle(blobImage, p1, p2, color, 1, 8, 0);
		if (labelled)
			{
			rectangle(blobImage, p1, p2, color, 1, 8, 0);
			putText(blobImage, label, p1, FONT_HERSHEY_SIMPLEX, 0.5, color);
			}
			else
				rectangle(blobImage, p1, p2, Scalar(255, 255, 255), 1, 8, 0);
	}

	//destroy all resources (if required)
	//...

	//return the image to show
	return blobImage;
}


/**
 *	Blob extraction from 1-channel image (binary). The extraction is performed based
 *	on the analysis of the connected components. All the input arguments must be 
 *  initialized when using this function.
 *
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image) 
 * \param bloblist List with found blobs
 *
 * \return Operation code (negative if not succesfull operation) 
 */
 int extractBlobs(cv::Mat fgmask, std::vector<cvBlob> &bloblist, int connectivity)
 {
	 //required variables for connected component analysis
	 //...
	 Mat aux; // image to be updated each time a blob is detected (blob cleared)
	 fgmask.convertTo(aux,CV_32SC1);

	 Mat mask;
	 mask.create(fgmask.rows + 2, fgmask.cols + 2, CV_8UC1);

	 //clear blob list (to fill with this function)
	 bloblist.clear();

	 aux = (aux / 255); // Converts the frame to have only 0 and 1. 0 - background and shadow, 1 - foreground
	 int ID = 0; // Variable for painting connected components
	 Rect r; // Variable for floodfill function


	 //Connected component analysis
	 for (int i = 0; i < aux.rows; i++)
	 {
		 for (int j = 0; j < aux.cols; j++)
		 {

			 if (aux.at<int>(i, j) == 1)
			 {
				 mask = Scalar::all(0);
				 ID += 1;
				 floodFill(aux, mask, Point(j, i),Scalar(1), &r, Scalar(), Scalar(), connectivity| cv::FLOODFILL_MASK_ONLY);
				 cvBlob blob = initBlob(ID, r.x, r.y, r.width, r.height);
				 bloblist.push_back(blob);

				 Range rows(1, mask.rows - 1);
				 Range columns(1, mask.cols - 1);
				 Mat new_mask = mask(rows, columns);
				 new_mask.convertTo(new_mask, CV_32SC1);
				 aux = aux.mul(1-new_mask);
			 }
		 }

	}
 }


int removeSmallBlobs(std::vector<cvBlob> bloblist_in, std::vector<cvBlob> &bloblist_out, int min_width, int min_height)
{

	//clear blob list (to fill with this function)
	bloblist_out.clear();


	for(int i = 0; i < bloblist_in.size(); i++)
	{
		cvBlob blob_in = bloblist_in[i]; //get ith blob
		if (blob_in.w > min_width&& blob_in.h > min_height) {
			bloblist_out.push_back(blob_in); // void implementation (does not remove)
		}
	}

	return 1;
}



 /**
  *	Blob classification between the available classes in 'Blob.hpp' (see CLASS typedef). All the input arguments must be
  *  initialized when using this function.
  *
  * \param frame Input image
  * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
  * \param bloblist List with found blobs
  *
  * \return Operation code (negative if not succesfull operation)
  */

 // ASPECT RATIO MODELS
#define MEAN_PERSON 0.3950
#define STD_PERSON 0.1887

#define MEAN_CAR 1.4736
#define STD_CAR 0.2329

#define MEAN_OBJECT 1.2111
#define STD_OBJECT 0.4470

// end ASPECT RATIO MODELS

// distances
float ED(float val1, float val2)
{
	return sqrt(pow(val1-val2,2));
}

float WED(float val1, float val2, float std)
{
	return sqrt(pow(val1-val2,2)/pow(std,2));
}
//end distances
 int classifyBlobs(std::vector<cvBlob> &bloblist, bool dist_metric)
 {
	 float dist, dist2, dist3;
	 for (int i = 0; i < bloblist.size(); i++)
	 {
		 cvBlob blob = bloblist[i]; //get i-th blob
		 float aspect_ratio = (float)blob.w / blob.h;

		 if(dist_metric)
		 {
			 //Code to classify blob into person, object, car and unknown using weighted euclidean distance
			 dist = WED(aspect_ratio, MEAN_PERSON, STD_PERSON);
			 dist2 = WED(aspect_ratio, MEAN_CAR, STD_CAR);
			 dist3 = WED(aspect_ratio, MEAN_OBJECT, STD_OBJECT);
			 if(dist<=3.5){
						 blob.label = PERSON;
					 }
			 if(dist2<=3){
				 if(dist2<=dist){
					 blob.label = OBJECT;
				 }
			 }
			 if (dist3<=3){
				 if(dist3<dist && dist3<dist2){
					 blob.label = CAR;
				 }
			 }
			 bloblist[i] = blob;
		 }
		 else
		 {
			 //Code to classify into person, object, car and unknown using euclidean distance
			dist = abs(aspect_ratio - MEAN_PERSON);
			dist2 = abs(aspect_ratio - MEAN_CAR);
			dist3 = abs(aspect_ratio - MEAN_OBJECT);

			if (dist <= 3.5 * STD_PERSON) {
				blob.label = PERSON;
				dist = abs(STD_PERSON - dist);
			}
			else if (dist2 <= 3 * STD_CAR)
			{
				dist2 = abs(STD_CAR - dist2);
				if (blob.label == UNKNOWN || dist2 > dist) {
					blob.label = CAR;
				}

			}
			else if (dist3 <= 3 * STD_OBJECT) {
				dist3 = abs(STD_OBJECT - dist3);
				if (blob.label == UNKNOWN) {
					blob.label = OBJECT;
				}
				else if ((blob.label == PERSON && dist3 > dist) || (blob.label == CAR && dist3 > dist2)) {
					blob.label = OBJECT;
				}


			}
			bloblist[i] = blob;
		 }

	 }
 	return 1;
 }

//stationary blob extraction function
 /**
  *	Stationary FG detection
  *
  * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
  * \param fgmask_history Foreground history counter image (1-channel integer image)
  * \param sfgmask Foreground/Background segmentation mask (1-channel binary image)
  *
  * \return Operation code (negative if not succesfull operation)
  *
  *
  * Based on: Stationary foreground detection for video-surveillance based on foreground and motion history images, D.Ortego, J.C.SanMiguel, AVSS2013
  *
  */

#define FPS 25 //check in video - not really critical
#define SECS_STATIONARY 1 // to set
#define I_COST 1 // to set // increment cost for stationarity detection
#define D_COST 1 // to set // decrement cost for stationarity detection
#define STAT_TH 0.8 // to set

 int extractStationaryFG (Mat fgmask, Mat &fgmask_history, Mat &sfgmask)
 {
	int numframes4static=(int)(FPS*SECS_STATIONARY);

	Mat aux,mask, norm_fg_hist;
	norm_fg_hist = Mat::zeros(Size(fgmask.cols, fgmask.rows), CV_32FC1);
	aux = (fgmask.clone()) / 255;
	aux.convertTo(aux, CV_32FC1); // has 1 for foreground pixels, 0 for background

	fgmask_history += (float)I_COST * aux - (float)D_COST*(1-aux); // Update foreground history
	norm_fg_hist = min(1, fgmask_history / numframes4static); //Normalization

	sfgmask = norm_fg_hist > STAT_TH; //Check for stationary threshold

 return 1;
 }









