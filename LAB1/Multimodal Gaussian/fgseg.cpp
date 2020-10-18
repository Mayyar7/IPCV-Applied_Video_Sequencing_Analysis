/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.cpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */

#include <opencv2/opencv.hpp>
#include "fgseg.hpp"


using namespace fgseg;
using namespace std;
using namespace cv;

//default constructor

bgs::bgs(double threshold, bool rgb, double alpha, bool selective_update, double threshold_ghost) :
	_threshold(threshold), _alpha(alpha),
	_selective_update(selective_update), _rgb(rgb), tau(threshold_ghost) {
}

//default destructor
bgs::~bgs(void)
{
}

//method to initialize bkg (first frame - hot start)
void bgs::init_bkg(cv::Mat Frame)
{
	if (!_rgb) {
		_bkg = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1); 
		_bkg = Frame.clone();
		cvtColor(_bkg, _bkg, COLOR_BGR2GRAY);

		mean = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1);
		mean = _bkg.clone();
		mean.convertTo(mean, CV_64F);

		mean2 = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
		randu(mean2, Scalar(0), Scalar(255));

		mean3 = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
		randu(mean3, Scalar(0), Scalar(255));

		weight = Mat::ones(Size(Frame.cols, Frame.rows), CV_64F);
		weight2 = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
		weight3 = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);

		variance = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
		cv::Mat var, mn;
		meanStdDev(_bkg, mn, var);
		variance = variance + var;

		variance2 = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
		randu(variance2, Scalar(10), var);

		variance3 = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
		randu(variance3, Scalar(10), var);

	}
	else{
		
		cout << "Colour currently not supported" << endl;
		exit(1);
	}

}

//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame)
{
	_frame = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
	_diff = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
	_bgsmask = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
	_frame = Frame.clone();

	cv::Mat weights[3] = { weight, weight2,weight3 };
	cv::Mat means[3] = { mean, mean2, mean3 };
	cv::Mat vars[3] = { variance, variance2, variance3 };
	double wth = 0.7;
	cv::Mat deviation[3]; // = { Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F) ,Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F) ,Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F) };
	sqrt(vars[0], deviation[0]);
	sqrt(vars[1], deviation[1]);
	sqrt(vars[2], deviation[2]);
	cv::Mat temp;

	if (!_rgb)
	{		
		// Single Gaussian
		cvtColor(_frame, _frame, COLOR_BGR2GRAY);
		
		cv::Mat float_frame;
		_frame.convertTo(float_frame, CV_64F);

		cv::Mat md = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F); //difference from the mean
		cv::Mat bd = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F); // difference from the maximum deviation allowed

		cv::Mat match[3] = { Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1) ,Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1) ,Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1) };
		cv::Mat float_match = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);

		cv::Mat Wth = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);

		/* Check if the current frame pixels belong to which gaussian
		Update the one that it belongs to and reduce the weight by normalization
		If it doesnt belong to any then remove the G with lowest weight and update it with mean as 
		the pixel value and the new deviation and smaller weight. Normalize it again*/
	

		for (int i = 0; i < 3; i++) {
			absdiff(means[i] , float_frame, md); // calculate the difference from the mean
			temp = _threshold * deviation[i]; // the boundary to check 
			absdiff(md, temp,bd); // calculate difference from the boundary
			match[i]  = (md <= temp)/255 ;// Gives 1 if the gaussian matches the pixel
			
			// Below lines to update the gaussian that detected the pixel as background 
			
			match[i].convertTo(float_match, CV_64F);
			means[i] = (_alpha * float_frame + (1 - _alpha) * means[i]).mul(float_match) + mean.mul(1 - float_match);
			vars[i] = (_alpha * ((float_frame - means[i]).mul(float_frame - means[i])) + (1 - _alpha) * vars[i]).mul(float_match) + vars[i].mul(1 - float_match);
			Wth += float_match.mul(weights[i]);
		}
		/*Step 0: Find the pixels that didn't belong to any of the exsiting Gaussians*/
		cv::Mat find_pix;
		threshold(match[0] + match[1] + match[2], find_pix, 0, 1,THRESH_BINARY); 
		//_bgsmask = find_pix.clone();
		_bgsmask = (Wth > wth) / 255;
		find_pix = 1 - find_pix;// This has now the mask 1 where it has no Gaussians matching

		//Step 1: Find least probable distribution from 3 for these pixels.
		// Will we check this with the weight/deviation value or the probability itself
		cv::Mat w1 = weights[0] / deviation[0];
		cv::Mat w2 = weights[1] / deviation[1];
		cv::Mat w3 = weights[2] / deviation[2];
		

		cv::Mat greater_list[3] = { Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1) ,Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1) ,Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1) };

		greater_list[0] += ((w1 < w2) / 255).mul(find_pix); // Compare 1>2
		greater_list[1] += ((w2 < w1) / 255).mul(find_pix); // Compare 2>1

		greater_list[0] += ((w1 < w3) / 255).mul(find_pix); // 1>3
		greater_list[2] += ((w3 < w1) / 255).mul(find_pix); // 3>1

		greater_list[1] += ((w2 < w3) / 255).mul(find_pix); //2>3
		greater_list[2] += ((w3 < w2) / 255).mul(find_pix); //3>2


		threshold(greater_list[0], greater_list[0], 1, 1, THRESH_BINARY); // Mask of the pixels in gauss 1 that needs to be updated
		threshold(greater_list[1], greater_list[1], 1, 1, THRESH_BINARY); // Mask of the pixels in gauss 2 that needs to be updated
		threshold(greater_list[2], greater_list[2], 1, 1, THRESH_BINARY); // Mask of the pixels in gauss 3 that needs to be updated

		cv::Mat greater_list_float[3];
		greater_list[0].convertTo(greater_list_float[0], CV_64F);
		greater_list[1].convertTo(greater_list_float[1], CV_64F);
		greater_list[2].convertTo(greater_list_float[2], CV_64F);
		//Step 2: Set mean as current pixel value, give it a high value of deviation
		//keep the weight low
		means[0] = float_frame.mul(greater_list_float[0]) + means[0].mul(1 - greater_list_float[0]);
		means[1] = float_frame.mul(greater_list_float[1]) + means[1].mul(1 - greater_list_float[1]);
		means[2] = float_frame.mul(greater_list_float[2]) + means[1].mul(1 - greater_list_float[2]);
		
		vars[0] = 4* vars[0].mul(greater_list_float[0]) + vars[0].mul(1 - greater_list_float[0]);
		vars[1] = 4 * vars[1].mul(greater_list_float[1]) + vars[1].mul(1 - greater_list_float[1]); // Double deviation = 4* variance as variance = deviation^2
		vars[2] = 4 * vars[2].mul(greater_list_float[1]) + vars[2].mul(1 - greater_list_float[2]);
		
		weights[0] = 0.3 * (weights[0] + 0.001).mul(greater_list_float[0]) + weights[0].mul(1 - greater_list_float[0]);
		weights[1] = 0.3 * (weights[1] + 0.001).mul(greater_list_float[1]) + weights[1].mul(1 - greater_list_float[1]); // decrease the weight by a factor of 0.3
		weights[2] = 0.3 * (weights[2] + 0.001).mul(greater_list_float[2]) + weights[2].mul(1 - greater_list_float[2]);
		

		//Now adjust the weights of all the gaussians based on the formula
		// on if they matched or not
		match[0].convertTo(float_match, CV_64F);
		weights[0] = (1 - _alpha) * weights[0] + _alpha * float_match;

		match[1].convertTo(float_match, CV_64F);
		weights[1] = (1 - _alpha) * weights[1] + _alpha * float_match;

		match[2].convertTo(float_match, CV_64F);
		weights[2] = (1 - _alpha) * weights[2] + _alpha * float_match;
		
		//Renormalize the weights.
		cv::Mat sum_val = weights[0] + weights[1] + weights[2];
		weights[0] /= sum_val;
		weights[1] /= sum_val;
		weights[2] /= sum_val;

		// Now with the updated Gaussians calculate the bgsmask


		

	}
	else{
		cout << "Colour currently not supported" << endl;
		exit(1);
	 }

}

//method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows()
{
	double thresh_h = 70;
	double thresh_s = 70;
	if (!_rgb){
		_shadowmask = Mat::zeros(Size(_bgsmask.cols,_bgsmask.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
		_fgmask = Mat::zeros(Size(_bgsmask.cols,_bgsmask.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
		_fgmask = (1 - _bgsmask)*255;
		_bgsmask *= 255;
		}

		else
	{
		cout << "Colour currently not supported" << endl;
		exit(1);
		_shadowmask = Mat::zeros(Size(_bgsmask.cols, _bgsmask.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
		_fgmask = Mat::zeros(Size(_bgsmask.cols, _bgsmask.rows), CV_8UC1); 

	

	 }


}
#if unused
void fgseg::bgs::getGaussPDF(cv::Mat img, cv::Mat& res)
{
	static const float inv_sqrt_2pi = 0.3989422804014327;
	cv::Mat a = Mat::zeros(Size(_bgsmask.cols, _bgsmask.rows), CV_64F);
	cv::subtract(img, mean, a);
	a = a / dn;
	cv::Mat b = Mat::zeros(Size(_bgsmask.cols, _bgsmask.rows), CV_64F);
	cv::exp(-0.5f * a.mul(a), b);
	res = (inv_sqrt_2pi / deviation).mul(b);
}

#endif // 0




//ADD ADDITIONAL FUNCTIONS HERE



