# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

* First, I converted the images to grayscale,
* Use  gaussian_blur method to deal with img
* Canny function to figure out edge
* Defined a vertices and find out region_of_interest
* Use hough transform find hough_lines and drawlines put into img.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by figure out the max and min X Y position in line. Then connect this two points.



If you'd like to include images to show how the pipeline works, here is how to include an image: 

![result](test_images_output\output solidYellowCurve2.jpg)


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when can find the correct max and min point.This line is not from top to the end.Shorter than it should be. 

Another shortcoming could be, test in video sometimes line didn't appear in correct position,it flys.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use fitting to find out the line. It would be a good way ,I am not familiar with this method, I can not finish it in a short time.I'll do it later.

Another potential improvement could be to set parameter in hough and region_of_interest automatically.Optional Challenge Video3,not finish yet.