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

bgs::bgs(double threshold, bool rgb, double alpha , bool selective_update , double threshold_ghost):
				_threshold(threshold),		_alpha(alpha),
				_selective_update(selective_update),		_rgb(rgb),  tau(threshold_ghost)
{
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

		variance = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
		cv::Mat var, mn;
		meanStdDev(_bkg, mn, var);

		variance = variance + var;
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
	cv::Mat deviation = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);
	if (!_rgb)
	{		
		// Single Gaussian
		cvtColor(_frame, _frame, COLOR_BGR2GRAY);
		
		cv::Mat float_frame;
		_frame.convertTo(float_frame, CV_64F);
		
		cv::sqrt(variance, deviation);
		// frame value should be between 
		// mean + 3*deviation and mean - 3*deviation
		cv::Mat lb = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F); 
		cv::Mat ub = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);

		lb = mean - _threshold * deviation;
		ub = mean + _threshold * deviation; // Lower bound and upper bound check

		cv::Mat bgsmask_float = Mat::zeros(Size(Frame.cols, Frame.rows), CV_64F);

		cv::Mat a = float_frame >= lb; // Intermediate logical values to check 
		cv::Mat b = float_frame <= ub;

		bitwise_and(a, b, _bgsmask);
		//_bgsmask = (float_frame >= lb)/255 + (float_frame <= ub)/255;
		//threshold(_bgsmask, _bgsmask, 1, 1, THRESH_BINARY);
		_bgsmask /= 255;
		
		_bgsmask.convertTo(bgsmask_float, CV_64F);
		mean = (_alpha * float_frame + (1 - _alpha) * mean).mul(bgsmask_float)+ mean.mul(1 - bgsmask_float);				
		variance = (_alpha * ((float_frame - mean).mul(float_frame - mean)) + (1 - _alpha) * variance).mul(bgsmask_float) +variance.mul(1 - bgsmask_float);


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