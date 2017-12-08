## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)




**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd code cell of the IPython notebook located in "Advanced Lane Finding Submission.ipynb" function Calibrate_Camera()

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

![CameraCalibration1](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/CameraCalibration1.png)
![CameraCalibration2](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/CameraCalibration2.png)

From above we can see that there are 3 images calibration1.jpg, calibration4.jpg and calibration5.jpg which do not have any corner lines connected to them because the number of specified corners are not being found in these images. So they are not used for camera calibration.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![undistort_output](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/undistort_output.png)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images

![Undistorted1](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/undistorted.png)

The effect of undistort is not so easy to recognize above, but can be perceived from the difference in shape of the hood of the car at the bottom corners of the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (code cell 6 function `Gradient_Threshold()`).  

Finally, I chose the sobel gradient thresholds horizontal x and direction. I chose the combination of Red and Green so the yellow lines can be detected properly. I used the S and L channel of the HLS colorspace. S channel because it detects well the bright yellow and white lines. L channel to avoid pixel values which have shadow. I fine tuned them and then combined all of them together.

Here's an example of my output for this step. 

![Gradient1](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/Gradient1.png)

![Graident2](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/Gradient2.png)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `GetPerspectiveTransformMatrix()`, which appears in lines code cell 10.  The `GetPerspectiveTransformMatrix()` function takes as source (`src`) and destination (`dst`) points and returns the Transform Matrix and inverse transfrom matrix.  I chose the hardcode the source and destination points in the following manner:

```python
def source():
    src = np.float32([
            [220,720],
            [1120,720],
            [570,470],
            [720,470]
        ])
    return src

def destination():
    dst = np.float32([
        [320,720],
        [920,720],
        [320,1],
        [920,1]
    ])
    return dst
    
def GetPerspectiveTransformMatrix(src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv
```

![Warped1](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/Warped1.png)

![Warped2](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/Warped2.png)

I applied the Region of Interest (ROI) or mask to my images which appears in code cell 8 in `apply_roi()`. The masked area values are hardcoded inside the funtion.
Following is the result you get after applying gradient and masking:

![Maksed1](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/Maksed1.png)

![Maksed2](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_imagess/Maksed2.png)

Following is the result you get after applying warping on the masked gradient images:

![Warped_Gradient1](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/Warped_Gradient1.png)

![Warped_Gradient2](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/Warped_Gradient2.png)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions `Initial_LineSearch()` and `Look_AheadSearch()`, identifies lane lines and fit a second order polynomial to both right and left lane lines, are present in code cell 18 and 26. The `Initial_LineSearch()` computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. These locations are identified from the local maxima of the left and right halves of the histogram. The function then identifies 10 windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image. Pixels belonging to each lane line are identified and the Numpy polyfit() method fits a second order polynomial to each set of pixels. The image below demonstrates how this process works:

![Sliding_Window](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/Sliding_Window.png)

Somehow the windowa didn't show up in my python notebook but appears fine in pycharm.

The `Look_AheadSearch()` finds the left and right lane line indices based on the previous polynomial fit values of left and right lines. The function performs the same task, but leverages by using a previous fit (from a previous video frame, for example) and only searches for lane pixels within a certain range of that fit.

The image below shows the histogram generated by `Look_AheadSearch()`; the resulting base points for the left and right lanes - the two peaks nearest the center - are clearly visible:

![Histogram](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/Histogram.png)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calcuated using `measure_radius_of_curvature()` in code cell 16 and center of offset is calculated using `Center_Offset()`. 

The radius of curvature is computed according to the formula and method described in the course material. Since we perform the polynomial fit in pixels and whereas the curvature has to be calculated in real world meters, we have to use a pixel to meter transformation and recompute the fit again.

The mean of the lane pixels closest to the car gives us the center of the lane. The center of the image gives us the position of the car. The difference between the 2 is the offset from the center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step inside `Pipeline()`.  

1. Paint the lane area
2. Perform an inverse perspective transform
3. Combine the processed image with the original image.

```python
left_line_window = np.array(np.transpose(np.vstack([left_x_predictions, y_points])))
    right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_x_predictions, y_points])))) 
    
    poly_points = np.vstack([left_line_window, right_line_window])
    
    out_img = np.dstack((warped, warped, warped))*255
    cv2.fillPoly(out_img, np.int_([poly_points]), [0,255, 0])
    
    unwarped = cv2.warpPerspective(out_img, Minv, img_size , flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)
    
    # compute the radius of curvature
    left_curve_rad = measure_radius_of_curvature(left_x_predictions)
    right_curve_rad = measure_radius_of_curvature(right_x_predictions)
    
    # compute the offset from the center
    center_offset = Center_Offset(img, left_x_predictions, right_x_predictions)
    result = DrawText(result, left_curve_rad, right_curve_rad, center_offset)
```

Here is an example of my result on a test image:

![Processed](https://github.com/vikasmalik22/Advanced_Lane_Finding/tree/master/examples/output_images/Processed.png)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any probl()ems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? 

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

## Problems
1. The most difficult part was to get the proper gradient image with the color thresholds. I spent most of my time experimenting here as the lane detection always used to fail during certain places like under tunnel or on very bright regions. Finally ecperimenting a lot and suggestions from others the Red and Green channel thresholds were able to help in detecting yellow color properly. Also, L channel threshold helped in finding the darker regions correctly.

2. Performing the Sanity checks correctly was also very challenging. To get the right combination of checks required lot of error and trials. To find out the bad frames I did the following:
	1. `verify_detectedlines()` which checks if the avarage detected lines difference was more than the previous computed values over a certain threshold. In this case I hardcode it to 100. 
	2. `cal_recentaverage_line()` this function calculated the average of the lines values over the last 10 frames. 
	3. `calc_mean_diff_lines()` this function was used to smooth the detected lane lines by taking the weightage of the previous and newly detected lines.

## Further Improvement
If I had more time, I would try to implement a even better binary image processing. I think better results can be achieved by trying more color models and channels. This would give me better results.

Current pipeline fails if assumptions are violated - e.g. turn is so steep that only one lane is visible, while we try to find both lines. Certain road colors can add false positives to the picture, causing pipeline to fail. Taking a better perspective transform can improve this. 

Also, for the sharper turns averaging over less number of last frames can give better results. Currently, I hardcoded this to the value of last 10 frames.