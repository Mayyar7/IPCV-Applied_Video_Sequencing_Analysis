# Applied_Video_Sequencing_Analysis

This repository containes routines for video analysis implemented as a part of IPCV Master program coursework. 

## LAB1: Foreground Segmentation
The code contains foreground segmentation algorithms and methods for single stationary cameras.
1. Frame Difference
2. Progressive Selective Update
3. Stationary Object Suppression
4. Shadown Detection
5. Unimodal Gaussian
6. Multimodal Gaussians

## LAB2: Object Detection and Classification
The code contains algorithms to extract BLOBobs (Binary Large OBject) for stationary cameras. Primarily the GrassFire Algorithm has been implemented to detect the connected regions in a the foreground segmentation mask. Two implementations of the GrassFire are available
1. Sequential Implementation
2. Recursive Implementation
<\br>
Morphological Operations like Opening and Closing have also been implemented to imrpove the foreground segmentation mask to fill in holes and remove noise. 

## LAB3: Kalman Filtering for Object Tracking
The code contains implementaiton of the Kalman Filter class of C++ to track objects in single stationary cameras for different videos. 

## LAB4: Histogram-based Object Tracking
The code contains histogram based algorithms for object tracking. Different types of histogram features have been used namely
1. Color Based Histogram Features
2. Gradient Based Histogram Features
3. Using both Color and Gradient histogram as Features
<\br>
Primarily the candidate generation function has been implemented as suggested by [[1]](#1). The algorithm is based on a square lattice grid around the location of the candidate for the object to be tracked along with a stride parameter that defines the search area for the candidate in the next frame. The size of grid is defined based on the number of candidates, and can only be square numbers.

## References
<a id="1">[1]</a> 
P. Fieguth and D. Terzopoulos,
Color-based tracking of heads and other mobile objects at video frame rates. 
Proceedings of IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 1997, pp. 21–27.
