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
bgs::bgs(double threshold, bool rgb, bool shadow_removal,double alpha, bool selective_update, double threshold_ghost):
_threshold(threshold), _shadow_removal(shadow_removal),_alpha(alpha), _selective_update(selective_update), _rgb(rgb), tau(threshold_ghost)
{
	_alpha_shadow = 0.5;
	_beta_shadow = 0.9;
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
		_counter = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1);
		_bkg = Frame.clone();
		cvtColor(_bkg, _bkg, COLOR_BGR2GRAY);
	}
	else{
		_bkg = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1);
		_counter = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1);
		_bkg = Frame.clone();
	}

}

//method to perform BackGroundSubtraction- Frame difference, Selective Update and Blind update
//bgsMask: 1=Foreground, 0 = Background

void bgs::bkgSubtraction(cv::Mat Frame)
{
	_frame = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
	_diff = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
	_bgsmask = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
	cv::Mat gray_diff = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1);
	_frame = Frame.clone();

	if (!_rgb)
	{
		cvtColor(_frame, _frame, COLOR_BGR2GRAY);
		if (_selective_update)
		{
			absdiff(_frame, _bkg, _diff);
			threshold(_diff, _bgsmask, _threshold, 1, THRESH_BINARY);

			cv::Mat new_bkg = (_alpha * _frame) + ((1 - _alpha) * _bkg);
			_bkg = new_bkg.mul((1-_bgsmask)) + _bkg.mul(_bgsmask);

		}

		else
		{
			// blind mode
			if(_shadow_removal == false)
			{
			_bkg = _alpha*_frame + (1-_alpha)*_bkg;
			}
		}

			absdiff(_frame, _bkg, _diff);
			threshold(_diff, _bgsmask, _threshold, 1, THRESH_BINARY);

			//Ghost suppression
			_counter = (_counter + _bgsmask).mul(_bgsmask);
			cv::Mat update_mat;
			threshold(_counter,update_mat,tau,1,THRESH_BINARY);
			_bkg = _bkg.mul((1 - update_mat)) + _frame.mul((update_mat));
			_counter = _counter.mul((1-update_mat));
			absdiff(_frame, _bkg, _diff);
			threshold(_diff, _bgsmask, _threshold, 1, THRESH_BINARY);
			_bgsmask *= 255;
	}
	//If it is a coloured frame
	else
	{
		if (_selective_update)
		{
			absdiff(_frame, _bkg, _diff);
			cvtColor(_diff, gray_diff, COLOR_BGR2GRAY);
			threshold(gray_diff, _bgsmask, _threshold, 1, THRESH_BINARY);
			cv::Mat f_channels[3];
			cv::Mat b_channels[3];
			split(_frame, f_channels);
			split(_bkg, b_channels);

			cv::Mat new_b[3];
			new_b[0] = (f_channels[0] * _alpha + (1 - _alpha) * b_channels[0]).mul((1 - _bgsmask)) + b_channels[0].mul(_bgsmask);
			new_b[1] = (f_channels[1] * _alpha + (1 - _alpha) * b_channels[1]).mul((1 - _bgsmask)) + b_channels[1].mul(_bgsmask);
			new_b[2] = (f_channels[2] * _alpha + (1 - _alpha) * b_channels[2]).mul((1 - _bgsmask)) + b_channels[2].mul(_bgsmask);

			//Removal of temporary ghosts
			_counter = (_counter + _bgsmask).mul(_bgsmask);
			cv::Mat update_mat;
			threshold(_counter, update_mat, tau, 1, THRESH_BINARY);
			new_b[0] = new_b[0].mul(1 - update_mat) + f_channels[0].mul(update_mat);
			new_b[1] = new_b[1].mul(1 - update_mat) + f_channels[1].mul(update_mat);
			new_b[2] = new_b[2].mul(1 - update_mat) + f_channels[2].mul(update_mat);
			_counter = _counter.mul((1 - update_mat));
			merge(new_b, 3, _bkg);

		}
		else
		{
			if(_shadow_removal == false)
			{
			// blind mode
			cv::Mat f_channels[3];
			cv::Mat b_channels[3];
			split(_frame, f_channels);
			split(_bkg, b_channels);
			b_channels[0] = f_channels[0] * _alpha + (1 - _alpha) * b_channels[0];
			b_channels[1] = f_channels[1] * _alpha + (1 - _alpha) * b_channels[1];
			b_channels[2] = f_channels[2] * _alpha + (1 - _alpha) * b_channels[2];
			merge(b_channels,3, _bkg);
			}

		}

		absdiff(_frame, _bkg, _diff);
		cvtColor(_diff, gray_diff, COLOR_BGR2GRAY);
		threshold(gray_diff, _bgsmask, _threshold, 1, THRESH_BINARY);
		_bgsmask = _bgsmask * 255;
	 }

}

//method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows()
{
	double thresh_h = 90;
	double thresh_s = 90;
	if (!_rgb)
	{
		_shadowmask = Mat::zeros(Size(_bgsmask.cols,_bgsmask.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
		_fgmask = Mat::zeros(Size(_bgsmask.cols,_bgsmask.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
		_bgsmask.copyTo(_fgmask);
	}

	else
	{
		_shadowmask = Mat::zeros(Size(_bgsmask.cols, _bgsmask.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
		_fgmask = Mat::zeros(Size(_bgsmask.cols, _bgsmask.rows), CV_8UC1);
		if(_shadow_removal == true)
		{
		//Converting frame to HSV
		cv::Mat _frameHSV = Mat::zeros(Size(_bgsmask.cols, _bgsmask.rows), CV_8UC1);
		cvtColor(_frame, _frameHSV, COLOR_BGR2HSV);

		//Converting HSV values to double
		cv::Mat float_frameHSV = Mat::zeros(Size(_bgsmask.cols, _bgsmask.rows),  CV_64F);
		_frameHSV.convertTo(float_frameHSV, CV_64F);

		//Converting bkg to HSV
		cv::Mat _bkgHSV = Mat::zeros(Size(_bgsmask.cols, _bgsmask.rows), CV_8UC1);
		cvtColor(_bkg, _bkgHSV, COLOR_BGR2HSV);

		//Converting HSV values to double
		cv::Mat float_bkgHSV = Mat::zeros(Size(_bgsmask.cols, _bgsmask.rows),  CV_64F);
		_bkgHSV.convertTo(float_bkgHSV, CV_64F);


		cv::Mat frame_channel[3];
		cv::Mat bkg_channel[3];
		split(float_frameHSV, frame_channel);
		split(float_bkgHSV, bkg_channel);

		cv::Mat h_diff;
		absdiff(frame_channel[0], bkg_channel[0], h_diff); // |IH - BH|
		cv::Mat D_h = cv::min(h_diff, 360 - h_diff); //Dh = min(|IH - BH|, 360 - |IH - BH|)
		cv::Mat div_v = frame_channel[2] / bkg_channel[2]; // IV/BV
		cv::Mat s_diff;
		absdiff(frame_channel[1], bkg_channel[1], s_diff); // |IS - BS|


		//Checking Conditions for Shadow Detection

		// check if div_v is >= to alpha
		cv::Mat alpha_mask;
		cv::compare(div_v, _alpha_shadow, alpha_mask, CMP_GT);
		alpha_mask = alpha_mask / 255.0;

		//First one with beta
		cv::Mat beta_mask;
		cv::compare(div_v, _beta_shadow, beta_mask, CMP_LE);
		beta_mask = beta_mask / 255.0;

		// check if all elements of div_v is less than s_diff
		cv::Mat less_mask;
		cv::compare(s_diff, thresh_s, less_mask, CMP_LE);
		less_mask = less_mask / 255.0;

		// check if dh is <= thresh_h
		cv::Mat dh_mask;
		cv::compare(D_h, thresh_h, dh_mask, CMP_LE);
		dh_mask = dh_mask/255.0;

		_shadowmask = alpha_mask.mul(beta_mask.mul(less_mask.mul(dh_mask)));
		_shadowmask = _shadowmask.mul(_bgsmask);//To ensure that only foreground pixels are included in show mask

		 //Shadow Removal
		_fgmask = _bgsmask.mul((1 - _shadowmask));
		}
		else{
			_bgsmask.copyTo(_fgmask);
		}

	 }


}
