## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[features1]: ./output_images/features1.png
[features2]: ./output_images/features2.png
[pipeline1]: ./output_images/pipeline1.png
[pipeline2]: ./output_images/pipeline2.png
[pipeline3]: ./output_images/pipeline3.png
[pipeline4]: ./output_images/pipeline4.png
[pipeline5]: ./output_images/pipeline5.png
[pipeline6]: ./output_images/pipeline6.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
**Here I will consider the rubric points individually and describe how I addressed each point in my implementation.**  
The code and referenced code cells can be found in the Jupyter Notebook accompanying the writeup.

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Code Cell No.: 5 + 6

I used the udacity training data containing two sets, namely `vehicle` and `non-vehicle` images. The data images are correlated because they were partly extracted from a video stream in sequential order creating a lot of very similar images, e.g., when the same car is extracted at different time steps throughout the training video data. For this reason, I manually separated the data in two categories, one for training data and one for test data. This simply involved moving approximately the first and the last 10% of images in each folder into the test image category making sure that the same car occurs only in one of the two sets of images.

I then explored different color spaces to use with `skimage.hog()` as well as to extract color features. By keeping other parameters fixed (e.g., `orientations`, `pixels_per_cell`, and `cells_per_block` of HOG) I trained a linear SVM classifier and recorded the test performance. The best performance was found using the `LUV` color space.

#### 2. Explain how you settled on your final choice of HOG parameters.

Once the `LUV` color space was settled for, I tried several different combinations of the other parameters via trial and error and incremental improvement. The parameters include: `orientations`, `pixels_per_cell`, and `cells_per_block` for HOG as well as what channels to use for HOG analysis, image size of spatial color features, and the number of bins for a color histogram of the image. The parameters that performed best on the test set as well as on the test video were chosen.

The final parameters are:  
* color space: LUV  

* hog channel: 0  
* orientations: 8  
* pixel_per_cell: 8  
* cells_per_block: 2  

* spatial size: 16x16  
* histogram bins: 32  

Here is an example of all features used according to these parameter values both for a car and non-car image:

![alt text][features1]
![alt text][features2]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Code Cell No.: 8 + 10

I trained a linear SVM on all data (training and test sets) with the previously chosen features: HOG features of L channel, spatial color features and histogram of colors. The features were normalized using the `StandardScaler` from `sklearn`. Additionally, I set the C parameter of SVM to 0.0001 to reduce overfitting as it showed to improve test performance.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Code Cell No.: 12

The sliding window search is done by first cutting off the upper part of the image that does not contain cars. In the lower part, all windows are searched and evaluated exhaustively with an overlap of 75% where one window is of size 64x64. Additionally, the image is scaled for a total of 5 times (factors: 5/7, 5/9, 5/11, 5/13, 1/3). This downscaling makes the 64x64 image region cover a larger area of the original image information. 

These choices worked well on the test video and were kept for this reason. It is possible to improve this procedure, e.g., by limiting the search area to where cars are expected or to reduce the number of scales to search so that fewer windows have to be evaluated which could speed up the pipeline.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The choice to use the LUV color space and HOG as well as color features led to a good result in classification. lowering the C parameter of the SVM classifier did the final trick in reducing overfitting.
Here are some example images:

![alt text][pipeline1]
![alt text][pipeline2]
![alt text][pipeline3]
![alt text][pipeline4]
![alt text][pipeline5]
![alt text][pipeline6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./videos/project_video_detection.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Code Cell No.: 14

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. In order to classify a positive detection as a car it has to occur in 8 subsequent frames. This is done by stacking the heatmaps (like a 3D-Tensor) and finding positions where positive detections touch across the 8 frames.
Once a vehicle is classified like this, I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I constructed bounding boxes to cover the area of each blob detected.
Detected vehicles do not have to constantly meet the requirement to occur across 8 frames. Known vehicle positions are updated by accepting all positive detections around the previous known position that is enlarged by 25% to each side.

To smooth the bounding boxes of the vehicles over several frames, exponential smoothing of the heatmaps is applied after false positives were filtered out. This is done by multiplying the last heatmap detection with 0.8 and adding the newly detected heatmap values weighted by 0.2 to it. Additionally, gaussian blurring is applied to the final heatmap which breaks the strict grid-like structure of the sliding window search and allows the detections to start/end at any intermediate pixel. The result is a reduction in the jitter-effect of subsequent positive detections and gives a smooth bounding box that follows the vehicles.

Finally, the pipeline is executed only on every second frame of the video to improve performance.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The designed pipeline and classifier might still overfit to the data given in this project. It works well on the given project video, but parameters were partly chosen so that it works for this video in particular. To bring this project to the next level, more video data would have to be analyzed in order to make it work across different lighting, road, car and noise conditions. Additionally, this pipeline might be too slow to work in a real-time environment. On my machine (laptop) it processes approximately one frame per second. One reasonable improvement would be to do a more selective evaluation of the image where window search is not done exhaustively, but rather on the interesting areas. For instance, new cars only appear at the right/left side of the image in a large scale (close to own car) or near the horizon in a smaller scale (far away of own car). Therefore, only two different scales could be used to find new cars whereas all scales are search at areas where a car was previously detected.

