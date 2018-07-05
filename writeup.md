## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog_features.png
[image3]: ./examples/window_sizes.png
[image4]: ./examples/hot_boxes.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fourth code cell of the Jupyter notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. The RGB colorspace allows lots of features with its three channels but I realized it also gave out lots of false positives in HOG space with the classifier. So I chose the GRAY-scale colorspace. The number of features were reduced but the output was more accurate. The processing time was also lesser as a result.

I did not take into account the same efficiency considerations, however, while choosing the number of gradient orientations per block. Although 8 oreintations gave decent enough results, to get better accuracy, I chose 16. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the `SVC()` function from `sklearn.svm` module. I extracted HOG features of each of the image in the traning set and stacked them as a single vector. The trainiing set was the verticle stack of HOG feaures of each of the sample images. I then normalized the HOG feature vectors using `StandardScaler()`. The label vector corresponded to 1, if car, and 0, if not car, to each sample in the verticle stack. Then I split the data into 80% training and 20% test data and fed it to the SVM.

The SVM is trained in the sixth code cell of the Jupyter notebook.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I chose three different window sizes. I started with 32x32, 64x64 and 128x128 and visualized the output on test images. Then after little experimenting, I settled upon the 80x80, 100x100 and 150x150 sizes as they looked visually sufficient for the expected scales of cars in the images. I chose 50% overlap to cover all windows from neighboring windows in all directions.

This windows selection is done in the seventh code cell of the Jupyter notebook. Here is the output of the chosen window sizes:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on just the grayscale images with 16 orientations.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  

I averaged the bounding boxes from the heatmap based on thresholds to get one bounding box for the detected vehicle. 

### Here's the result of six frames showing the heatmap of detections within a number of bounding boxes and then the average bounding box for the detection:

![alt text][image5]


In video processing, in order to avoid false positives, I also used averages from previous 10 frames so that the output is more robust.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This has been a tricky project. So many knobs to tune. There was a lot of experimentation. And it was also quite time consuming having to train a classifer many times with so many different parameters. In the end, the video conversion process was also extremely time consuming. Just to filter out the a few flickers of false positives, the processing had to be done multiple times, taking a lot of time.

Ideally, my response should be to try more colorspaces, more window sizes with differnt dimensions, extend the feature set etc in order to have such a robust classifier that would eliminate the need for filtering during video processing. But if I were to improve on this project with the objective of detecting cars, I would just use a deep neural net. Or at least do some GPU programming to run SVM, and even rest of the code, in GPU. I beleive I can already do that with tensorflow. So I would just use that if I were to train a SVM with further parameters.
 

